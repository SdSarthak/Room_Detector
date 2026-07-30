"""Building a labelled training set out of a directory of CSI captures.

Expected layout -- one directory per room, any number of captures inside::

    data/
      kitchen/    capture_01.csv  capture_02.csv
      bedroom/    capture_01.csv
      hallway/    capture_01.csv

A flat layout also works, where the label is the part of the filename before
the first underscore (``data/kitchen_01.csv``).

Each capture contributes several windows. Because windows cut from the same
capture are highly correlated, the returned ``groups`` array identifies the
source capture so that cross-validation can keep them in the same fold --
without that, accuracy scores are optimistic to the point of being meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .csi import read_csi_file, records_to_arrays
from .features import extract_features, feature_names
from .preprocessing import CSIPreprocessor

__all__ = ["CaptureFile", "Dataset", "discover_captures", "build_dataset"]

CSI_GLOB = "*.csv"


@dataclass(frozen=True)
class CaptureFile:
    path: Path
    label: str


@dataclass
class Dataset:
    """A materialised training set."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    captures: list[CaptureFile]
    preprocessor: CSIPreprocessor
    skipped: list[tuple[Path, str]]

    @property
    def labels(self) -> list[str]:
        return sorted(set(self.y.tolist()))

    def summary(self) -> str:
        counts = {label: int(np.sum(self.y == label)) for label in self.labels}
        lines = [
            f"{self.X.shape[0]} windows x {self.X.shape[1]} features "
            f"from {len(self.captures)} captures across {len(self.labels)} rooms",
        ]
        for label, count in sorted(counts.items()):
            lines.append(f"  {label:<20} {count:>6} windows")
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} capture(s):")
            for path, reason in self.skipped:
                lines.append(f"    {path.name}: {reason}")
        return "\n".join(lines)


def discover_captures(data_dir: str | Path) -> list[CaptureFile]:
    """Find every labelled capture under ``data_dir``."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {data_dir}. Record captures with "
            f"`python -m room_detector collect --room <name>` first."
        )

    captures: list[CaptureFile] = []
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for path in sorted(sub.glob(CSI_GLOB)):
            captures.append(CaptureFile(path=path, label=sub.name))

    if not captures:
        for path in sorted(data_dir.glob(CSI_GLOB)):
            label = path.stem.split("_")[0]
            if label:
                captures.append(CaptureFile(path=path, label=label))

    if not captures:
        raise FileNotFoundError(
            f"no CSI captures (*.csv) found under {data_dir}. Expected one "
            f"sub-directory per room, e.g. {data_dir / 'kitchen' / 'capture_01.csv'}"
        )
    return captures


def build_dataset(config: Config | None = None, data_dir: str | Path | None = None) -> Dataset:
    """Load every capture, clean it, and cut it into labelled feature windows."""

    config = config or Config()
    captures = discover_captures(data_dir if data_dir is not None else config.data_dir)

    # ---- pass 1: decode captures ------------------------------------------
    decoded: list[tuple[CaptureFile, dict[str, np.ndarray]]] = []
    skipped: list[tuple[Path, str]] = []
    for capture in captures:
        try:
            records = read_csi_file(capture.path)
        except (OSError, ValueError) as exc:
            skipped.append((capture.path, str(exc)))
            continue
        if len(records) < config.features.window_size:
            skipped.append(
                (
                    capture.path,
                    f"only {len(records)} usable packets, need at least "
                    f"{config.features.window_size}",
                )
            )
            continue
        decoded.append((capture, records_to_arrays(records)))

    if not decoded:
        detail = "; ".join(f"{path.name}: {reason}" for path, reason in skipped)
        raise ValueError(f"no capture yielded enough CSI packets to train on. {detail}")

    # ---- fit the preprocessor on all captures at once ---------------------
    # A subcarrier only counts as null when it is null in *every* capture, so
    # the per-capture column maxima are a sufficient (and tiny) summary.
    widths = {arrays["amplitude"].shape[1] for _, arrays in decoded}
    if len(widths) > 1:
        raise ValueError(
            f"captures disagree on subcarrier count ({sorted(widths)}). Re-flash the "
            f"ESP32 boards with a single CSI configuration and recapture."
        )
    summary = np.vstack([np.abs(arrays["amplitude"]).max(axis=0) for _, arrays in decoded])
    preprocessor = CSIPreprocessor(config.preprocess).fit(summary)

    # ---- pass 2: clean and window -----------------------------------------
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[int] = []
    kept_captures: list[CaptureFile] = []

    for capture, arrays in decoded:
        amplitude, phase = preprocessor.transform(arrays["amplitude"], arrays["phase"])
        try:
            features = extract_features(
                amplitude,
                phase,
                arrays["rssi"],
                arrays["noise_floor"],
                config.features,
            )
        except ValueError as exc:
            skipped.append((capture.path, str(exc)))
            continue

        group_id = len(kept_captures)
        kept_captures.append(capture)
        feature_rows.append(features)
        labels.extend([capture.label] * features.shape[0])
        groups.extend([group_id] * features.shape[0])

    if not feature_rows:
        raise ValueError("no capture produced a full feature window")

    X = np.vstack(feature_rows)
    y = np.asarray(labels, dtype=object)
    names = feature_names(preprocessor.n_subcarriers, config.features)
    if len(names) != X.shape[1]:  # pragma: no cover - guards a coding mistake
        raise AssertionError(
            f"feature name count ({len(names)}) does not match feature count ({X.shape[1]})"
        )

    return Dataset(
        X=X,
        y=y,
        groups=np.asarray(groups, dtype=int),
        feature_names=names,
        captures=kept_captures,
        preprocessor=preprocessor,
        skipped=skipped,
    )
