"""Cleaning of raw CSI amplitude and phase matrices.

Raw ESP32 CSI is noisy in three specific ways that all matter for room
classification:

1. Guard-band and DC subcarriers carry no information and are constant zero.
2. Individual packets contain large impulsive outliers (AGC steps, collisions).
3. The measured phase contains a linear ramp across subcarriers caused by
   sampling time offset and carrier frequency offset, which changes packet to
   packet even when the device has not moved.

The functions below address each, and :class:`CSIPreprocessor` chains them
according to a :class:`~room_detector.config.PreprocessConfig`.
"""

from __future__ import annotations

import numpy as np

from .config import PreprocessConfig

__all__ = [
    "LLTF_SUBCARRIERS",
    "LLTF_DATA_SUBCARRIERS",
    "select_subcarriers",
    "null_subcarrier_mask",
    "hampel_filter",
    "moving_average",
    "sanitize_phase",
    "normalize",
    "CSIPreprocessor",
]

#: Subcarrier count of a 20 MHz LLTF capture (128 raw bytes / 2).
LLTF_SUBCARRIERS = 64

#: Indices of the 52 data subcarriers of an 802.11 20 MHz LLTF symbol.
#: Bins 0..5, 32 and 59..63 are guard band, DC and edge bins.
LLTF_DATA_SUBCARRIERS: tuple[int, ...] = tuple(range(6, 32)) + tuple(range(33, 59))


def select_subcarriers(matrix: np.ndarray, indices: "np.ndarray | tuple[int, ...] | None" = None) -> np.ndarray:
    """Keep only ``indices`` columns of a ``(n_packets, n_subcarriers)`` matrix.

    Defaults to :data:`LLTF_DATA_SUBCARRIERS`. If the matrix does not have the
    expected 64 columns (for instance an HT-LTF capture) it is returned
    unchanged, since the LLTF layout would not apply.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_2d(matrix)
    if indices is None:
        if matrix.shape[1] != LLTF_SUBCARRIERS:
            return matrix
        indices = LLTF_DATA_SUBCARRIERS
    indices = np.asarray(indices, dtype=int)
    if indices.size and (indices.min() < 0 or indices.max() >= matrix.shape[1]):
        raise IndexError(
            f"subcarrier index out of range for a matrix with {matrix.shape[1]} subcarriers"
        )
    return matrix[:, indices]


def null_subcarrier_mask(matrix: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of subcarriers that are (near) zero in *every* packet."""

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_2d(matrix)
    return np.all(np.abs(matrix) <= tol, axis=0)


#: Packets processed per Hampel pass. The filter needs a
#: ``(chunk, n_subcarriers, window)`` temporary, and ``np.median`` copies it, so
#: an unchunked pass over a long capture peaks at roughly
#: ``window * 4`` times the size of the capture itself.
_HAMPEL_CHUNK = 8192


def hampel_filter(matrix: np.ndarray, window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace impulsive outliers along the time axis with the local median.

    A value is an outlier when it deviates from the median of its temporal
    neighbourhood by more than ``n_sigmas`` robust standard deviations (derived
    from the median absolute deviation).

    Processed in chunks of :data:`_HAMPEL_CHUNK` packets so peak memory stays
    bounded regardless of capture length; the result is identical to filtering
    the whole matrix at once because each chunk is padded with the ``half``
    neighbouring packets it needs.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_2d(matrix)
    if window < 1 or matrix.shape[0] < 3:
        return matrix.copy()

    half = max(1, int(window) // 2)
    span = 2 * half + 1
    n_packets = matrix.shape[0]
    padded = np.pad(matrix, ((half, half), (0, 0)), mode="edge")
    cleaned = matrix.copy()

    for start in range(0, n_packets, _HAMPEL_CHUNK):
        stop = min(start + _HAMPEL_CHUNK, n_packets)
        # padded is offset by `half`, so rows [start, stop) need padded rows
        # [start, stop + 2*half) to form their full neighbourhoods.
        block = padded[start : stop + 2 * half]
        # Sliding view of shape (stop - start, n_subcarriers, span).
        strided = np.lib.stride_tricks.sliding_window_view(block, span, axis=0)
        medians = np.median(strided, axis=-1)
        mad = np.median(np.abs(strided - medians[:, :, None]), axis=-1)
        # 1.4826 makes the MAD a consistent estimator of sigma for normal data.
        sigma = 1.4826 * mad

        # Where sigma is 0 the neighbourhood is constant; only a differing value
        # is an outlier, and that is exactly what np.abs(...) > 0 catches.
        outliers = np.abs(matrix[start:stop] - medians) > (n_sigmas * sigma)
        # Basic slicing gives a view, so this writes through to `cleaned`.
        cleaned[start:stop][outliers] = medians[outliers]

    return cleaned


def moving_average(matrix: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth each subcarrier along time with a centred moving average."""

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_2d(matrix)
    if window <= 1 or matrix.shape[0] < 2:
        return matrix.copy()

    window = min(int(window), matrix.shape[0])
    half = window // 2
    padded = np.pad(matrix, ((half, window - 1 - half), (0, 0)), mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return strided.mean(axis=-1)


def sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """Remove the linear phase ramp caused by timing and frequency offsets.

    For each packet the phase is unwrapped across subcarriers and the best fit
    line ``a * k + b`` is subtracted. This is the standard linear phase
    calibration used in WiFi sensing: it discards the offset terms that vary
    randomly per packet while preserving the multipath structure that actually
    identifies a room.
    """

    phase = np.asarray(phase, dtype=np.float64)
    _check_2d(phase)
    n_subcarriers = phase.shape[1]
    if n_subcarriers < 2:
        return phase.copy()

    unwrapped = np.unwrap(phase, axis=1)
    k = np.arange(n_subcarriers, dtype=np.float64)
    k_centered = k - k.mean()
    denominator = np.sum(k_centered**2)
    if denominator == 0:  # pragma: no cover - needs a single subcarrier
        return unwrapped

    slope = (unwrapped * k_centered).sum(axis=1, keepdims=True) / denominator
    intercept = unwrapped.mean(axis=1, keepdims=True) - slope * k.mean()
    return unwrapped - (slope * k + intercept)


def normalize(matrix: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalise each packet (row) independently.

    ``l2`` divides by the row norm, which cancels transmit-power and
    automatic-gain differences. ``zscore`` and ``minmax`` rescale each row to
    zero mean / unit variance and to ``[0, 1]`` respectively.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_2d(matrix)
    if method == "none":
        return matrix.copy()
    if method == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)
    if method == "zscore":
        mean = matrix.mean(axis=1, keepdims=True)
        std = matrix.std(axis=1, keepdims=True)
        return (matrix - mean) / np.where(std == 0, 1.0, std)
    if method == "minmax":
        low = matrix.min(axis=1, keepdims=True)
        high = matrix.max(axis=1, keepdims=True)
        span = high - low
        return (matrix - low) / np.where(span == 0, 1.0, span)
    raise ValueError(f"unknown normalisation method: {method!r}")


class CSIPreprocessor:
    """Applies the configured cleaning steps to amplitude and phase matrices.

    The set of kept subcarriers is decided on the first call and reused
    afterwards, so a model trained on a capture keeps the same feature layout
    when it later runs against a live stream.
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.subcarrier_indices: np.ndarray | None = None

    def fit(self, amplitude: np.ndarray) -> "CSIPreprocessor":
        """Decide which subcarriers to keep from a representative capture."""

        amplitude = np.asarray(amplitude, dtype=np.float64)
        _check_2d(amplitude)
        indices = np.arange(amplitude.shape[1])

        if self.config.use_data_subcarriers_only and amplitude.shape[1] == LLTF_SUBCARRIERS:
            indices = np.asarray(LLTF_DATA_SUBCARRIERS, dtype=int)

        if self.config.drop_null_subcarriers:
            nulls = null_subcarrier_mask(amplitude[:, indices])
            if not nulls.all():  # never drop everything
                indices = indices[~nulls]

        self.subcarrier_indices = indices
        return self

    def transform(
        self, amplitude: np.ndarray, phase: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Clean ``amplitude`` (and ``phase``) using the fitted subcarrier set."""

        amplitude = np.asarray(amplitude, dtype=np.float64)
        _check_2d(amplitude)
        if self.subcarrier_indices is None:
            raise RuntimeError("CSIPreprocessor.fit() must be called before transform()")

        indices = self.subcarrier_indices
        if indices.size and indices.max() >= amplitude.shape[1]:
            raise ValueError(
                f"preprocessor was fitted on captures with at least {int(indices.max()) + 1} "
                f"subcarriers but got {amplitude.shape[1]}"
            )

        amp = amplitude[:, indices]
        if self.config.hampel_window >= 1:
            amp = hampel_filter(amp, self.config.hampel_window, self.config.hampel_sigmas)
        if self.config.smooth_window > 1:
            amp = moving_average(amp, self.config.smooth_window)
        if self.config.normalize != "none":
            amp = normalize(amp, self.config.normalize)

        cleaned_phase: np.ndarray | None = None
        if phase is not None:
            phase = np.asarray(phase, dtype=np.float64)
            _check_2d(phase)
            if phase.shape != amplitude.shape:
                raise ValueError(
                    f"phase shape {phase.shape} does not match amplitude shape {amplitude.shape}"
                )
            cleaned_phase = phase[:, indices]
            if self.config.sanitize_phase:
                cleaned_phase = sanitize_phase(cleaned_phase)
            if self.config.smooth_window > 1:
                cleaned_phase = moving_average(cleaned_phase, self.config.smooth_window)

        return amp, cleaned_phase

    def fit_transform(
        self, amplitude: np.ndarray, phase: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return self.fit(amplitude).transform(amplitude, phase)

    @property
    def n_subcarriers(self) -> int:
        if self.subcarrier_indices is None:
            raise RuntimeError("CSIPreprocessor has not been fitted")
        return int(self.subcarrier_indices.size)


def _check_2d(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D (packets, subcarriers) matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"matrix is empty (shape {matrix.shape})")
