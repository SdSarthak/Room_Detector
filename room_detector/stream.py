"""Live room detection from a serial port, stdin or a capture file.

The ESP32 prints one CSI line per received packet. This module buffers those
lines into windows, applies exactly the preprocessing the model was trained
with, and emits a smoothed room prediction.
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

from .config import Config
from .csi import CSIRecord, iter_csi_records, records_to_arrays
from .features import window_feature_vector
from .model import RoomClassifier

__all__ = [
    "Prediction",
    "serial_lines",
    "stdin_lines",
    "file_lines",
    "RealtimeRoomDetector",
    "run_stream",
]


@dataclass(frozen=True)
class Prediction:
    """One smoothed room prediction over a window of packets."""

    room: str
    confidence: float
    raw_room: str
    raw_confidence: float
    n_packets: int
    timestamp: float

    def __str__(self) -> str:
        return (
            f"[{self.timestamp:>12.3f}] room={self.room:<16} "
            f"confidence={self.confidence:.3f} (window vote of {self.n_packets} packets)"
        )


# ----------------------------------------------------------------------
# Line sources
# ----------------------------------------------------------------------
def serial_lines(port: str, baudrate: int = 921600, timeout: float = 1.0) -> Iterator[str]:
    """Yield decoded lines from an ESP32 serial port. Requires ``pyserial``.

    A read that times out yields an empty string rather than looping silently,
    so a caller enforcing a deadline (``collect --seconds``) regains control
    even when the board is sending nothing at all. Every consumer in this
    package already skips empty lines.
    """

    try:
        import serial
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pyserial is required to read from a serial port; install it with "
            "`pip install pyserial`"
        ) from exc

    try:
        connection = serial.Serial(port, baudrate, timeout=timeout)
    except serial.SerialException as exc:
        raise ConnectionError(
            f"could not open serial port {port!r} at {baudrate} baud: {exc}. "
            f"Check the cable and that no other program (e.g. a serial monitor) holds the port."
        ) from exc

    with connection:
        while True:
            try:
                raw = connection.readline()
            except serial.SerialException as exc:  # pragma: no cover - hardware dependent
                raise ConnectionError(f"serial port {port!r} disconnected: {exc}") from exc
            if not raw:
                yield ""  # read timeout: hand control back to the caller
                continue
            yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def stdin_lines() -> Iterator[str]:
    """Yield lines from stdin, tolerating the non UTF-8 bytes ESP32 boots emit."""

    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            return
        yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def file_lines(path: str | Path) -> Iterator[str]:
    """Replay a recorded capture file, line by line."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"capture not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


# ----------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------
class RealtimeRoomDetector:
    """Sliding-window room detection over a stream of :class:`CSIRecord`.

    The windowing and preprocessing settings always come from the config the
    *model was trained with*, never from ``config``. Only the stream settings
    (serial port, vote window, minimum confidence) are taken from the caller.
    This matters because the feature vector has the same length for every
    ``window_size`` -- the features are per-subcarrier statistics -- so
    classifying 16-packet windows with a model trained on 64-packet windows
    raises no error at all, it just returns confident nonsense.
    """

    def __init__(self, model: RoomClassifier, config: Config | None = None) -> None:
        self.model = model
        training = model.config
        if config is None or config is training:
            self.config = training
        else:
            if config.features != training.features or config.preprocess != training.preprocess:
                print(
                    "note: ignoring the current window/preprocessing settings and using the "
                    f"ones the model was trained with (window_size="
                    f"{training.features.window_size}, window_step="
                    f"{training.features.window_step}). Retrain if you want to change them.",
                    file=sys.stderr,
                )
            self.config = replace(
                config, features=training.features, preprocess=training.preprocess
            )
        self.preprocessor = model.preprocessor
        if self.preprocessor is None:
            raise ValueError(
                "the loaded model has no subcarrier configuration attached; retrain it "
                "so live predictions use the same features as training"
            )
        window = self.config.features.window_size
        self._buffer: deque[CSIRecord] = deque(maxlen=window)
        self._votes: deque[tuple[str, float]] = deque(maxlen=self.config.stream.vote_window)
        self._since_last = 0
        self.packets_seen = 0
        #: Windows actually run through the classifier, whether or not the
        #: smoothed prediction cleared ``stream.min_confidence``.
        self.windows_classified = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._votes.clear()
        self._since_last = 0

    def feed(self, record: CSIRecord) -> Prediction | None:
        """Add one packet; return a prediction when a full window is ready."""

        self._buffer.append(record)
        self.packets_seen += 1
        self._since_last += 1

        if len(self._buffer) < self._buffer.maxlen:
            return None
        if self._since_last < self.config.features.window_step:
            return None
        self._since_last = 0
        return self._classify_buffer()

    def _classify_buffer(self) -> Prediction | None:
        records = list(self._buffer)
        arrays = records_to_arrays(records)
        if arrays["amplitude"].shape[0] < self.config.features.window_size:
            # Mixed subcarrier widths inside one window; wait for a clean one.
            return None

        amplitude, phase = self.preprocessor.transform(arrays["amplitude"], arrays["phase"])
        features = window_feature_vector(
            amplitude, phase, arrays["rssi"], arrays["noise_floor"], self.config.features
        )
        raw_room, raw_confidence = self.model.predict_one(features)
        self.windows_classified += 1

        self._votes.append((raw_room, raw_confidence))
        room, confidence = _majority_vote(self._votes)
        if confidence < self.config.stream.min_confidence:
            return None

        return Prediction(
            room=room,
            confidence=confidence,
            raw_room=raw_room,
            raw_confidence=raw_confidence,
            n_packets=len(records),
            timestamp=float(arrays["timestamp"][-1]),
        )

    def process(self, lines: Iterable[str]) -> Iterator[Prediction]:
        """Consume raw serial lines and yield predictions as they are produced."""

        for record in iter_csi_records(lines):
            prediction = self.feed(record)
            if prediction is not None:
                yield prediction


def _majority_vote(votes: "deque[tuple[str, float]] | list[tuple[str, float]]") -> tuple[str, float]:
    """Most frequent room in the vote buffer, with its mean confidence.

    Ties are broken by total confidence so the result is deterministic.
    """

    votes = list(votes)
    if not votes:
        raise ValueError("cannot vote on an empty prediction buffer")

    counts = Counter(room for room, _ in votes)
    totals: dict[str, float] = {}
    for room, confidence in votes:
        totals[room] = totals.get(room, 0.0) + confidence

    best = max(counts, key=lambda room: (counts[room], totals[room]))
    return best, float(totals[best] / counts[best])


def run_stream(
    model: RoomClassifier,
    lines: Iterable[str],
    config: Config | None = None,
    on_prediction=None,
    detector: RealtimeRoomDetector | None = None,
) -> int:
    """Drive a detector over ``lines``. Returns the number of predictions made.

    Pass ``detector`` to keep a handle on it -- its counters explain an empty
    result (no full window vs. every window suppressed by ``min_confidence``).
    """

    if detector is None:
        detector = RealtimeRoomDetector(model, config)
    emitted = 0
    for prediction in detector.process(lines):
        emitted += 1
        if on_prediction is None:
            print(prediction, flush=True)
        else:
            on_prediction(prediction)
    return emitted
