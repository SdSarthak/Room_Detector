"""Room detection from ESP32 WiFi Channel State Information.

Typical use::

    from room_detector import Config, build_dataset, RoomClassifier

    config = Config()
    dataset = build_dataset(config)
    model = RoomClassifier(config).fit_dataset(dataset)
    print(model.evaluate(dataset.X, dataset.y, dataset.groups))
    model.save()
"""

from .config import (
    Config,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    StreamConfig,
)
from .csi import (
    CSIParseError,
    CSIRecord,
    iter_csi_records,
    parse_csi_line,
    read_csi_file,
    read_csi_files,
    records_to_arrays,
)
from .dataset import Dataset, build_dataset, discover_captures
from .features import extract_features, feature_names, window_feature_vector
from .model import EvaluationResult, RoomClassifier, build_estimator
from .preprocessing import CSIPreprocessor, sanitize_phase
from .stream import Prediction, RealtimeRoomDetector

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "PreprocessConfig",
    "FeatureConfig",
    "ModelConfig",
    "StreamConfig",
    "CSIRecord",
    "CSIParseError",
    "parse_csi_line",
    "iter_csi_records",
    "read_csi_file",
    "read_csi_files",
    "records_to_arrays",
    "CSIPreprocessor",
    "sanitize_phase",
    "window_feature_vector",
    "extract_features",
    "feature_names",
    "Dataset",
    "build_dataset",
    "discover_captures",
    "RoomClassifier",
    "EvaluationResult",
    "build_estimator",
    "RealtimeRoomDetector",
    "Prediction",
]
