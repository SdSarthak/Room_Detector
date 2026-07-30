"""Configuration for the room detection pipeline.

Every knob lives here so that nothing downstream needs a hardcoded path or a
magic number. Values may be supplied programmatically, loaded from a YAML file
or taken from the environment (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "PreprocessConfig",
    "FeatureConfig",
    "ModelConfig",
    "StreamConfig",
    "Config",
]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={raw!r} is not an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name}={raw!r} is not a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PreprocessConfig:
    """Controls how raw CSI is turned into a clean amplitude/phase matrix."""

    #: Keep only the 802.11 20 MHz data subcarriers instead of all 64 bins.
    use_data_subcarriers_only: bool = True
    #: Drop subcarriers that are identically zero across the whole capture.
    drop_null_subcarriers: bool = True
    #: Hampel outlier filter window (in packets). 0 disables the filter.
    hampel_window: int = 5
    #: Hampel rejection threshold, in robust standard deviations.
    hampel_sigmas: float = 3.0
    #: Moving-average smoothing window (in packets). 0 or 1 disables it.
    smooth_window: int = 3
    #: Remove the linear phase ramp caused by sampling/timing offsets.
    sanitize_phase: bool = True
    #: Per-packet amplitude normalisation: "none", "l2", "zscore" or "minmax".
    normalize: str = "none"

    def __post_init__(self) -> None:
        allowed = {"none", "l2", "zscore", "minmax"}
        if self.normalize not in allowed:
            raise ValueError(f"normalize must be one of {sorted(allowed)}, got {self.normalize!r}")
        if self.hampel_window < 0:
            raise ValueError("hampel_window must be >= 0")
        if self.smooth_window < 0:
            raise ValueError("smooth_window must be >= 0")


@dataclass
class FeatureConfig:
    """Controls the sliding window and which statistics become features."""

    #: Number of consecutive packets that make up one classified sample.
    window_size: int = 64
    #: Packets to advance between consecutive windows.
    window_step: int = 32
    #: Include per-subcarrier amplitude statistics (mean/std/median/...).
    use_amplitude: bool = True
    #: Include sanitized per-subcarrier phase statistics.
    use_phase: bool = True
    #: Include RSSI / noise-floor summary statistics.
    use_rssi: bool = True
    #: Include first-difference ("motion energy") statistics.
    use_motion: bool = True

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.window_step < 1:
            raise ValueError("window_step must be >= 1")
        if not (self.use_amplitude or self.use_phase or self.use_rssi):
            raise ValueError("at least one of use_amplitude/use_phase/use_rssi must be enabled")


@dataclass
class ModelConfig:
    """Classifier selection and training hyper-parameters."""

    #: One of "random_forest", "extra_trees", "svm", "knn", "logistic".
    algorithm: str = "random_forest"
    n_estimators: int = 300
    max_depth: int | None = None
    #: SVM / logistic regularisation strength.
    C: float = 10.0
    #: KNN neighbour count.
    n_neighbors: int = 5
    random_state: int = 42
    #: Folds used by ``evaluate``. Grouped by capture file to avoid leakage.
    cv_folds: int = 5

    def __post_init__(self) -> None:
        allowed = {"random_forest", "extra_trees", "svm", "knn", "logistic"}
        if self.algorithm not in allowed:
            raise ValueError(f"algorithm must be one of {sorted(allowed)}, got {self.algorithm!r}")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")


@dataclass
class StreamConfig:
    """Serial / live-prediction settings."""

    port: str = "COM3"
    baudrate: int = 921600
    timeout: float = 1.0
    #: Majority vote over this many consecutive window predictions.
    vote_window: int = 3
    #: Emit a prediction only when the smoothed confidence clears this bar.
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.vote_window < 1:
            raise ValueError("vote_window must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


@dataclass
class Config:
    """Top level configuration object."""

    data_dir: Path = Path("data")
    model_path: Path = Path("models/room_classifier.joblib")
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.model_path = Path(self.model_path)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a config from environment variables (see ``.env.example``)."""

        load_dotenv()
        return cls(
            data_dir=Path(_env("ROOM_DETECTOR_DATA_DIR", "data")),
            model_path=Path(_env("ROOM_DETECTOR_MODEL_PATH", "models/room_classifier.joblib")),
            preprocess=PreprocessConfig(
                use_data_subcarriers_only=_env_bool("ROOM_DETECTOR_DATA_SUBCARRIERS_ONLY", True),
                drop_null_subcarriers=_env_bool("ROOM_DETECTOR_DROP_NULL_SUBCARRIERS", True),
                hampel_window=_env_int("ROOM_DETECTOR_HAMPEL_WINDOW", 5),
                hampel_sigmas=_env_float("ROOM_DETECTOR_HAMPEL_SIGMAS", 3.0),
                smooth_window=_env_int("ROOM_DETECTOR_SMOOTH_WINDOW", 3),
                sanitize_phase=_env_bool("ROOM_DETECTOR_SANITIZE_PHASE", True),
                normalize=_env("ROOM_DETECTOR_NORMALIZE", "none"),
            ),
            features=FeatureConfig(
                window_size=_env_int("ROOM_DETECTOR_WINDOW_SIZE", 64),
                window_step=_env_int("ROOM_DETECTOR_WINDOW_STEP", 32),
                use_amplitude=_env_bool("ROOM_DETECTOR_USE_AMPLITUDE", True),
                use_phase=_env_bool("ROOM_DETECTOR_USE_PHASE", True),
                use_rssi=_env_bool("ROOM_DETECTOR_USE_RSSI", True),
                use_motion=_env_bool("ROOM_DETECTOR_USE_MOTION", True),
            ),
            model=ModelConfig(
                algorithm=_env("ROOM_DETECTOR_ALGORITHM", "random_forest"),
                n_estimators=_env_int("ROOM_DETECTOR_N_ESTIMATORS", 300),
                C=_env_float("ROOM_DETECTOR_C", 10.0),
                n_neighbors=_env_int("ROOM_DETECTOR_N_NEIGHBORS", 5),
                random_state=_env_int("ROOM_DETECTOR_RANDOM_STATE", 42),
                cv_folds=_env_int("ROOM_DETECTOR_CV_FOLDS", 5),
            ),
            stream=StreamConfig(
                port=_env("ROOM_DETECTOR_SERIAL_PORT", "COM3"),
                baudrate=_env_int("ROOM_DETECTOR_BAUDRATE", 921600),
                timeout=_env_float("ROOM_DETECTOR_SERIAL_TIMEOUT", 1.0),
                vote_window=_env_int("ROOM_DETECTOR_VOTE_WINDOW", 3),
                min_confidence=_env_float("ROOM_DETECTOR_MIN_CONFIDENCE", 0.0),
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Build a config from a nested mapping, ignoring unknown keys."""

        sections = {
            "preprocess": PreprocessConfig,
            "features": FeatureConfig,
            "model": ModelConfig,
            "stream": StreamConfig,
        }
        kwargs: dict[str, Any] = {}
        for key in ("data_dir", "model_path"):
            if key in data:
                kwargs[key] = Path(data[key])
        for name, section_cls in sections.items():
            raw = data.get(name) or {}
            valid = {f.name for f in fields(section_cls)}
            unknown = set(raw) - valid
            if unknown:
                raise ValueError(f"unknown {name} option(s): {sorted(unknown)}")
            kwargs[name] = section_cls(**raw)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a config from a YAML file. Requires ``pyyaml``."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "pyyaml is required to load YAML configs; install it with `pip install pyyaml`"
            ) from exc

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"config file {path} must contain a mapping at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["model_path"] = str(self.model_path)
        return data


def load_dotenv(path: str | Path = ".env") -> bool:
    """Load ``KEY=VALUE`` pairs from a dotenv file into ``os.environ``.

    Existing environment variables always win. Returns ``True`` if a file was
    read. Implemented locally so the package has no hard dependency on
    ``python-dotenv``.
    """

    path = Path(path)
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return True
