"""The room classifier: a scikit-learn pipeline plus persistence.

The saved artifact bundles the estimator *and* the preprocessing decisions
(which subcarriers were kept, which window size was used). Without that bundle
a model is unusable, because a live stream must be reduced to exactly the same
feature layout it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config, ModelConfig
from .preprocessing import CSIPreprocessor

__all__ = ["build_estimator", "EvaluationResult", "RoomClassifier"]

_MODEL_FORMAT_VERSION = 1


def build_estimator(config: ModelConfig | None = None) -> Pipeline:
    """Create the scaler + classifier pipeline described by ``config``."""

    config = config or ModelConfig()
    if config.algorithm == "random_forest":
        clf: Any = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
    elif config.algorithm == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
    elif config.algorithm == "svm":
        clf = SVC(C=config.C, kernel="rbf", gamma="scale", probability=True,
                  random_state=config.random_state)
    elif config.algorithm == "knn":
        clf = KNeighborsClassifier(n_neighbors=config.n_neighbors, weights="distance")
    elif config.algorithm == "logistic":
        clf = LogisticRegression(C=config.C, max_iter=2000, random_state=config.random_state)
    else:  # pragma: no cover - ModelConfig validates this
        raise ValueError(f"unknown algorithm: {config.algorithm!r}")

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


@dataclass
class EvaluationResult:
    """Cross-validated scores for a trained model."""

    accuracy: float
    labels: list[str]
    confusion: np.ndarray
    report: str
    n_samples: int
    n_folds: int
    grouped: bool

    def __str__(self) -> str:
        scheme = "grouped by capture" if self.grouped else "stratified"
        header = (
            f"{self.n_folds}-fold cross-validation ({scheme}) over {self.n_samples} windows\n"
            f"accuracy: {self.accuracy:.4f}\n"
        )
        width = max((len(label) for label in self.labels), default=4)
        matrix = ["confusion matrix (rows = true, cols = predicted):"]
        matrix.append(" " * (width + 2) + " ".join(f"{label[:6]:>6}" for label in self.labels))
        for label, row in zip(self.labels, self.confusion):
            matrix.append(f"  {label:<{width}}" + " ".join(f"{value:>6d}" for value in row))
        return header + "\n".join(matrix) + "\n\n" + self.report


class RoomClassifier:
    """Trains, evaluates, persists and applies a room classification model."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []
        self.feature_names: list[str] = []
        self.preprocessor: CSIPreprocessor | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        preprocessor: CSIPreprocessor | None = None,
    ) -> "RoomClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} labels")
        if len(set(y.tolist())) < 2:
            raise ValueError(
                "need captures from at least two rooms to train a classifier; "
                f"found only {sorted(set(y.tolist()))}"
            )

        self.pipeline = build_estimator(self.config.model)
        self.pipeline.fit(X, y)
        self.classes_ = [str(c) for c in self.pipeline.classes_]
        self.feature_names = list(feature_names) if feature_names else []
        self.preprocessor = preprocessor
        return self

    def fit_dataset(self, dataset: "Any") -> "RoomClassifier":
        """Train directly from a :class:`~room_detector.dataset.Dataset`."""

        return self.fit(dataset.X, dataset.y, dataset.feature_names, dataset.preprocessor)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _require_fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("model is not trained; call fit() or load() first")
        return self.pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._require_fitted().predict(self._as_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pipeline = self._require_fitted()
        if not hasattr(pipeline, "predict_proba"):  # pragma: no cover - all current models support it
            raise AttributeError(f"{self.config.model.algorithm} does not expose probabilities")
        return pipeline.predict_proba(self._as_matrix(X))

    def predict_one(self, features: np.ndarray) -> tuple[str, float]:
        """Classify a single feature vector, returning ``(room, confidence)``."""

        proba = self.predict_proba(np.asarray(features).reshape(1, -1))[0]
        index = int(np.argmax(proba))
        return self.classes_[index], float(proba[index])

    def _as_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        expected = getattr(self._require_fitted(), "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"model expects {expected} features per sample but got {X.shape[1]}. "
                f"The capture was probably recorded with a different window size or "
                f"subcarrier configuration than the training data."
            )
        return X

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> EvaluationResult:
        """Cross-validate a *fresh* estimator on ``X``/``y``.

        When ``groups`` is given, ``GroupKFold`` keeps all windows cut from one
        capture inside a single fold. Otherwise neighbouring windows overlap
        across the train/test split and the accuracy is inflated.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        labels = sorted(set(y.tolist()))
        if len(labels) < 2:
            raise ValueError("need at least two rooms to evaluate a classifier")

        folds = min(self.config.model.cv_folds, _max_folds(y, groups))
        if folds < 2:
            raise ValueError(
                "not enough captures/samples per room to cross-validate; "
                "record at least two captures per room"
            )

        estimator = build_estimator(self.config.model)
        grouped = groups is not None
        if grouped:
            splitter: Any = GroupKFold(n_splits=folds)
            predictions = cross_val_predict(estimator, X, y, groups=groups, cv=splitter)
        else:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                       random_state=self.config.model.random_state)
            predictions = cross_val_predict(estimator, X, y, cv=splitter)

        return EvaluationResult(
            accuracy=float(accuracy_score(y, predictions)),
            labels=labels,
            confusion=confusion_matrix(y, predictions, labels=labels),
            report=classification_report(y, predictions, labels=labels, zero_division=0),
            n_samples=int(X.shape[0]),
            n_folds=folds,
            grouped=grouped,
        )

    def feature_importances(self, top: int = 20) -> list[tuple[str, float]]:
        """Most informative features, when the estimator exposes importances."""

        pipeline = self._require_fitted()
        classifier = pipeline.named_steps["classifier"]
        importances = getattr(classifier, "feature_importances_", None)
        if importances is None:
            return []
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
        return [(name, float(value)) for name, value in ranked[:top]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """Persist the pipeline together with its preprocessing decisions."""

        import joblib

        pipeline = self._require_fitted()
        path = Path(path) if path is not None else self.config.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": _MODEL_FORMAT_VERSION,
                "pipeline": pipeline,
                "classes": self.classes_,
                "feature_names": self.feature_names,
                "config": self.config.to_dict(),
                "subcarrier_indices": (
                    None if self.preprocessor is None else self.preprocessor.subcarrier_indices
                ),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RoomClassifier":
        import joblib

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"model not found: {path}. Train one with `python -m room_detector train`."
            )
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError(f"{path} is not a room_detector model bundle")
        version = bundle.get("format_version")
        if version != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by model format v{version}, this build expects "
                f"v{_MODEL_FORMAT_VERSION}. Retrain the model."
            )

        model = cls(Config.from_dict(bundle.get("config", {})))
        model.pipeline = bundle["pipeline"]
        model.classes_ = list(bundle.get("classes", []))
        model.feature_names = list(bundle.get("feature_names", []))
        indices = bundle.get("subcarrier_indices")
        if indices is not None:
            preprocessor = CSIPreprocessor(model.config.preprocess)
            preprocessor.subcarrier_indices = np.asarray(indices, dtype=int)
            model.preprocessor = preprocessor
        return model


def _max_folds(y: np.ndarray, groups: np.ndarray | None) -> int:
    """Largest usable fold count for this data."""

    if groups is not None:
        # GroupKFold cannot make more folds than there are distinct captures.
        return int(len(set(np.asarray(groups).tolist())))
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())
