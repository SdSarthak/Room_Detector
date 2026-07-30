"""Parsing of the CSV/serial output produced by the ESP32-CSI-Tool firmware.

Each CSI packet is emitted by ``_components/csi_component.h`` as a single line::

    CSI_DATA,<role>,<mac>,<rssi>,...,<len>,[<i0> <r0> <i1> <r1> ... ]

There are 25 scalar fields followed by a bracketed array of signed bytes. The
array interleaves the *imaginary* and *real* part of every subcarrier, in that
order -- matching ``ESP32-CSI-Tool/python_utils/parse_csi.py``.

Two details make a naive ``line.split(",")`` unsafe in practice:

* The ``len`` field reports the length the radio *would* have produced. When
  the firmware is built with ``CONFIG_SHOULD_COLLECT_ONLY_LLTF`` it truncates
  the array to 128 bytes but still reports 384, so the declared length cannot
  be trusted -- always use the array actually present.
* Serial output interleaves boot/debug chatter with CSI lines and a line can be
  cut in half by a reset, so parsing must tolerate garbage and truncation.
"""

from __future__ import annotations

import csv as _csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "CSI_MARKER",
    "CSI_COLUMNS",
    "CSIParseError",
    "CSIRecord",
    "parse_csi_line",
    "iter_csi_records",
    "read_csi_file",
    "read_csi_files",
    "records_to_arrays",
    "raw_to_complex",
]

#: Token that marks the beginning of a CSI line in the serial stream.
CSI_MARKER = "CSI_DATA"

#: Scalar column names, in firmware order (``_print_csi_csv_header``).
CSI_COLUMNS: tuple[str, ...] = (
    "role",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "real_time_set",
    "real_timestamp",
    "declared_len",
)

#: Number of comma separated fields before the bracketed CSI array.
_N_SCALAR_FIELDS = len(CSI_COLUMNS) + 1  # + the leading "CSI_DATA" marker

_ARRAY_RE = re.compile(r"\[([^\]]*)\]")

#: Integer valued columns, used to coerce the scalar fields.
_INT_COLUMNS = frozenset(CSI_COLUMNS) - {"role", "mac", "real_timestamp"}


class CSIParseError(ValueError):
    """Raised when a line cannot be interpreted as a CSI packet."""


@dataclass(frozen=True)
class CSIRecord:
    """A single decoded CSI packet."""

    role: str
    mac: str
    rssi: int
    rate: int
    sig_mode: int
    mcs: int
    bandwidth: int
    smoothing: int
    not_sounding: int
    aggregation: int
    stbc: int
    fec_coding: int
    sgi: int
    noise_floor: int
    ampdu_cnt: int
    channel: int
    secondary_channel: int
    local_timestamp: int
    ant: int
    sig_len: int
    rx_state: int
    real_time_set: int
    real_timestamp: float
    declared_len: int
    csi_raw: np.ndarray

    @property
    def n_subcarriers(self) -> int:
        """Number of subcarriers actually present in this packet."""

        return len(self.csi_raw) // 2

    @property
    def imaginary(self) -> np.ndarray:
        return self.csi_raw[0::2].astype(np.float64)

    @property
    def real(self) -> np.ndarray:
        return self.csi_raw[1::2].astype(np.float64)

    def complex(self) -> np.ndarray:
        """Per-subcarrier complex channel response."""

        return self.real + 1j * self.imaginary

    def amplitude(self) -> np.ndarray:
        """Per-subcarrier magnitude ``sqrt(imag^2 + real^2)``."""

        return np.hypot(self.imaginary, self.real)

    def phase(self) -> np.ndarray:
        """Per-subcarrier phase ``atan2(imag, real)`` in radians."""

        return np.arctan2(self.imaginary, self.real)


def raw_to_complex(csi_raw: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert an interleaved ``[imag, real, ...]`` buffer to complex values."""

    raw = np.asarray(csi_raw, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("csi_raw must be one dimensional")
    if raw.size % 2 != 0:
        raise ValueError(f"csi_raw must hold an even number of values, got {raw.size}")
    return raw[1::2] + 1j * raw[0::2]


def parse_csi_line(line: str, *, expected_subcarriers: int | None = None) -> CSIRecord:
    """Parse a single ``CSI_DATA`` line into a :class:`CSIRecord`.

    Leading serial noise before the ``CSI_DATA`` marker is discarded. Raises
    :class:`CSIParseError` for anything that is not a complete, well formed
    packet.
    """

    if not isinstance(line, str):
        raise CSIParseError(f"expected a string, got {type(line).__name__}")

    marker_at = line.find(CSI_MARKER)
    if marker_at < 0:
        raise CSIParseError("line does not contain a CSI_DATA marker")
    line = line[marker_at:].strip()

    match = _ARRAY_RE.search(line)
    if match is None:
        # A '[' with no closing ']' means the line was cut short mid transmission.
        raise CSIParseError("line has no complete bracketed CSI array")

    header = line[: match.start()].rstrip(",").rstrip()
    fields = header.split(",")
    if len(fields) != _N_SCALAR_FIELDS:
        raise CSIParseError(
            f"expected {_N_SCALAR_FIELDS} scalar fields before the CSI array, got {len(fields)}"
        )

    values: dict[str, object] = {}
    for name, raw_value in zip(CSI_COLUMNS, fields[1:]):
        raw_value = raw_value.strip()
        if name in _INT_COLUMNS:
            try:
                values[name] = int(raw_value)
            except ValueError as exc:
                raise CSIParseError(f"field {name!r} is not an integer: {raw_value!r}") from exc
        elif name == "real_timestamp":
            try:
                values[name] = float(raw_value)
            except ValueError as exc:
                raise CSIParseError(f"field 'real_timestamp' is not a number: {raw_value!r}") from exc
        else:
            values[name] = raw_value

    try:
        csi_raw = np.array(
            [int(token) for token in match.group(1).split() if token],
            dtype=np.int16,
        )
    except ValueError as exc:
        raise CSIParseError(f"CSI array holds a non-integer token: {exc}") from exc

    if csi_raw.size == 0:
        raise CSIParseError("CSI array is empty")
    if csi_raw.size % 2 != 0:
        raise CSIParseError(
            f"CSI array must hold an even number of values (imag/real pairs), got {csi_raw.size}"
        )
    if expected_subcarriers is not None and csi_raw.size // 2 != expected_subcarriers:
        raise CSIParseError(
            f"expected {expected_subcarriers} subcarriers, got {csi_raw.size // 2}"
        )

    return CSIRecord(csi_raw=csi_raw, **values)  # type: ignore[arg-type]


def iter_csi_records(
    lines: Iterable[str],
    *,
    strict: bool = False,
    expected_subcarriers: int | None = None,
) -> Iterator[CSIRecord]:
    """Yield a :class:`CSIRecord` for every parsable line in ``lines``.

    With ``strict=False`` (the default) unparsable lines are skipped, which is
    what you want for a live serial feed. With ``strict=True`` the first bad
    line raises :class:`CSIParseError`.
    """

    for line in lines:
        if not line or CSI_MARKER not in line:
            continue
        try:
            yield parse_csi_line(line, expected_subcarriers=expected_subcarriers)
        except CSIParseError:
            if strict:
                raise


def read_csi_file(
    source: str | Path | IO[str],
    *,
    strict: bool = False,
    expected_subcarriers: int | None = None,
) -> list[CSIRecord]:
    """Read every CSI packet from a capture file (or open file object)."""

    if hasattr(source, "read"):
        return list(
            iter_csi_records(source, strict=strict, expected_subcarriers=expected_subcarriers)  # type: ignore[arg-type]
        )

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"CSI capture not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(
            iter_csi_records(handle, strict=strict, expected_subcarriers=expected_subcarriers)
        )


def read_csi_files(
    paths: Iterable[str | Path],
    *,
    strict: bool = False,
    expected_subcarriers: int | None = None,
) -> list[CSIRecord]:
    """Read and concatenate several capture files."""

    records: list[CSIRecord] = []
    for path in paths:
        records.extend(
            read_csi_file(path, strict=strict, expected_subcarriers=expected_subcarriers)
        )
    return records


def records_to_arrays(records: Sequence[CSIRecord]) -> dict[str, np.ndarray]:
    """Stack a list of records into aligned arrays.

    Packets whose subcarrier count differs from the most common one are dropped,
    since the firmware can switch between LLTF-only and HT-LTF captures mid
    stream and the two are not comparable.

    Returns a dict with ``amplitude`` and ``phase`` matrices of shape
    ``(n_packets, n_subcarriers)`` plus 1-D ``rssi``, ``noise_floor`` and
    ``timestamp`` arrays.
    """

    if not records:
        raise ValueError("no CSI records to stack")

    counts = np.array([record.n_subcarriers for record in records])
    modal = int(np.bincount(counts).argmax())
    kept = [record for record in records if record.n_subcarriers == modal]

    amplitude = np.vstack([record.amplitude() for record in kept])
    phase = np.vstack([record.phase() for record in kept])
    return {
        "amplitude": amplitude,
        "phase": phase,
        "rssi": np.array([record.rssi for record in kept], dtype=np.float64),
        "noise_floor": np.array([record.noise_floor for record in kept], dtype=np.float64),
        "timestamp": np.array([record.real_timestamp for record in kept], dtype=np.float64),
        "n_dropped": np.array(len(records) - len(kept)),
    }


def write_csi_csv(path: str | Path, lines: Iterable[str]) -> int:
    """Persist raw serial lines to ``path``, writing the firmware CSV header.

    Returns the number of CSI lines written.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(("type",) + CSI_COLUMNS[:-1] + ("len", "CSI_DATA"))
        for line in lines:
            if CSI_MARKER not in line:
                continue
            handle.write(line.rstrip("\r\n") + "\n")
            written += 1
    return written
