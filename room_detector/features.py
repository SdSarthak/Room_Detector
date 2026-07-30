"""Turning a window of CSI packets into a fixed length fingerprint vector.

A single packet is far too noisy to identify a room. Instead a sliding window
of consecutive packets is summarised into per-subcarrier statistics: the
*mean* captures the static multipath signature of a location, the *variance*
and *motion* terms capture how the channel fluctuates there, and RSSI captures
coarse distance to the transmitter.
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig

__all__ = [
    "AMPLITUDE_STATS",
    "PHASE_STATS",
    "sliding_windows",
    "window_feature_vector",
    "extract_features",
    "feature_names",
    "FeatureExtractor",
]

#: Statistics computed per subcarrier over an amplitude window.
AMPLITUDE_STATS: tuple[str, ...] = ("mean", "std", "median", "min", "max", "iqr")

#: Statistics computed per subcarrier over a sanitized phase window.
PHASE_STATS: tuple[str, ...] = ("mean", "std")


def sliding_windows(n_packets: int, window_size: int, step: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs covering ``n_packets``.

    Yields nothing when the capture is shorter than a single window, so a short
    or truncated capture is skipped rather than padded into a fake sample.
    """

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    return [
        (start, start + window_size)
        for start in range(0, n_packets - window_size + 1, step)
    ]


def _stat_block(window: np.ndarray, stats: tuple[str, ...]) -> list[np.ndarray]:
    """Compute ``stats`` per column of a ``(window_size, n_subcarriers)`` block."""

    out: list[np.ndarray] = []
    for stat in stats:
        if stat == "mean":
            out.append(window.mean(axis=0))
        elif stat == "std":
            out.append(window.std(axis=0))
        elif stat == "median":
            out.append(np.median(window, axis=0))
        elif stat == "min":
            out.append(window.min(axis=0))
        elif stat == "max":
            out.append(window.max(axis=0))
        elif stat == "iqr":
            q75, q25 = np.percentile(window, [75, 25], axis=0)
            out.append(q75 - q25)
        else:  # pragma: no cover - guarded by the module level constants
            raise ValueError(f"unknown statistic: {stat!r}")
    return out


def window_feature_vector(
    amplitude: np.ndarray,
    phase: np.ndarray | None = None,
    rssi: np.ndarray | None = None,
    noise_floor: np.ndarray | None = None,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Summarise one window of packets into a 1-D feature vector.

    ``amplitude`` and ``phase`` are ``(window_size, n_subcarriers)`` matrices;
    ``rssi`` and ``noise_floor`` are ``(window_size,)`` vectors.
    """

    config = config or FeatureConfig()
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.size == 0:
        raise ValueError(f"amplitude must be a non-empty 2-D window, got shape {amplitude.shape}")

    parts: list[np.ndarray] = []

    if config.use_amplitude:
        parts.extend(_stat_block(amplitude, AMPLITUDE_STATS))

    if config.use_motion:
        # Mean absolute packet-to-packet change: how "alive" the channel is.
        if amplitude.shape[0] > 1:
            motion = np.abs(np.diff(amplitude, axis=0)).mean(axis=0)
        else:
            motion = np.zeros(amplitude.shape[1])
        parts.append(motion)

    if config.use_phase:
        if phase is None:
            raise ValueError("config.use_phase is enabled but no phase window was supplied")
        phase = np.asarray(phase, dtype=np.float64)
        if phase.shape != amplitude.shape:
            raise ValueError(
                f"phase window shape {phase.shape} does not match amplitude {amplitude.shape}"
            )
        parts.extend(_stat_block(phase, PHASE_STATS))

    if config.use_rssi:
        if rssi is None:
            raise ValueError("config.use_rssi is enabled but no rssi window was supplied")
        rssi = np.asarray(rssi, dtype=np.float64).ravel()
        if rssi.size != amplitude.shape[0]:
            raise ValueError(
                f"rssi window has {rssi.size} entries but the amplitude window has "
                f"{amplitude.shape[0]} packets"
            )
        scalars = [rssi.mean(), rssi.std(), rssi.min(), rssi.max()]
        if noise_floor is not None:
            noise_floor = np.asarray(noise_floor, dtype=np.float64).ravel()
            if noise_floor.size != amplitude.shape[0]:
                raise ValueError(
                    f"noise_floor window has {noise_floor.size} entries but the amplitude "
                    f"window has {amplitude.shape[0]} packets"
                )
            scalars.extend([noise_floor.mean(), rssi.mean() - noise_floor.mean()])
        else:
            scalars.extend([0.0, 0.0])
        parts.append(np.asarray(scalars, dtype=np.float64))

    return np.concatenate(parts)


def feature_names(n_subcarriers: int, config: FeatureConfig | None = None) -> list[str]:
    """Names matching, position for position, :func:`window_feature_vector`."""

    config = config or FeatureConfig()
    names: list[str] = []
    if config.use_amplitude:
        for stat in AMPLITUDE_STATS:
            names.extend(f"amp_{stat}_sc{i}" for i in range(n_subcarriers))
    if config.use_motion:
        names.extend(f"amp_motion_sc{i}" for i in range(n_subcarriers))
    if config.use_phase:
        for stat in PHASE_STATS:
            names.extend(f"phase_{stat}_sc{i}" for i in range(n_subcarriers))
    if config.use_rssi:
        names.extend(
            [
                "rssi_mean",
                "rssi_std",
                "rssi_min",
                "rssi_max",
                "noise_floor_mean",
                "snr_mean",
            ]
        )
    return names


def extract_features(
    amplitude: np.ndarray,
    phase: np.ndarray | None = None,
    rssi: np.ndarray | None = None,
    noise_floor: np.ndarray | None = None,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Slide a window over a whole capture and stack the feature vectors.

    Returns an ``(n_windows, n_features)`` matrix. Raises ``ValueError`` when
    the capture holds fewer packets than one window.
    """

    config = config or FeatureConfig()
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2:
        raise ValueError(f"amplitude must be 2-D, got shape {amplitude.shape}")

    bounds = sliding_windows(amplitude.shape[0], config.window_size, config.window_step)
    if not bounds:
        raise ValueError(
            f"capture holds {amplitude.shape[0]} packets, fewer than the "
            f"window_size of {config.window_size}"
        )

    rows = [
        window_feature_vector(
            amplitude[start:stop],
            phase[start:stop] if phase is not None else None,
            rssi[start:stop] if rssi is not None else None,
            noise_floor[start:stop] if noise_floor is not None else None,
            config,
        )
        for start, stop in bounds
    ]
    return np.vstack(rows)


class FeatureExtractor:
    """Convenience wrapper binding a :class:`FeatureConfig` to the helpers."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray | None = None,
        rssi: np.ndarray | None = None,
        noise_floor: np.ndarray | None = None,
    ) -> np.ndarray:
        return extract_features(amplitude, phase, rssi, noise_floor, self.config)

    def names(self, n_subcarriers: int) -> list[str]:
        return feature_names(n_subcarriers, self.config)
