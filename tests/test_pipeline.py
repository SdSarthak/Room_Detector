"""End-to-end tests: dataset building, training, persistence and live streaming."""

import numpy as np
import pytest

from room_detector.config import Config
from room_detector.dataset import build_dataset, discover_captures
from room_detector.model import RoomClassifier, build_estimator
from room_detector.simulate import room_channel, simulate_lines, write_simulated_capture
from room_detector.stream import RealtimeRoomDetector, _majority_vote, run_stream


@pytest.fixture(scope="module")
def config(simulated_data_dir):
    config = Config(data_dir=simulated_data_dir)
    config.features.window_size = 50
    config.features.window_step = 25
    config.model.n_estimators = 60
    return config


@pytest.fixture(scope="module")
def dataset(config):
    return build_dataset(config)


@pytest.fixture(scope="module")
def trained(config, dataset):
    return RoomClassifier(config).fit_dataset(dataset)


# ----------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------
def test_room_channel_is_stable_but_room_specific():
    np.testing.assert_allclose(room_channel("kitchen"), room_channel("kitchen"))
    assert not np.allclose(room_channel("kitchen"), room_channel("bedroom"))


def test_simulated_lines_round_trip_through_the_parser(sample_lines):
    from room_detector.csi import parse_csi_line

    assert len(sample_lines) == 80
    record = parse_csi_line(sample_lines[0])
    assert record.n_subcarriers == 64
    assert -128 <= record.csi_raw.min() and record.csi_raw.max() <= 127


def test_simulated_guard_subcarriers_are_null(sample_lines):
    from room_detector.csi import iter_csi_records, records_to_arrays

    arrays = records_to_arrays(list(iter_csi_records(sample_lines)))
    assert arrays["amplitude"][:, 32].max() == 0  # DC bin
    assert arrays["amplitude"][:, 20].max() > 0   # a data bin


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def test_discover_captures_labels_by_directory(simulated_data_dir):
    captures = discover_captures(simulated_data_dir)
    assert len(captures) == 9
    assert {capture.label for capture in captures} == {"kitchen", "bedroom", "hallway"}


def test_discover_captures_supports_a_flat_layout(tmp_path):
    write_simulated_capture(tmp_path / "kitchen_01.csv", "kitchen", 10)
    write_simulated_capture(tmp_path / "bedroom_01.csv", "bedroom", 10)
    assert {c.label for c in discover_captures(tmp_path)} == {"kitchen", "bedroom"}


def test_discover_captures_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_captures(tmp_path / "does_not_exist")


def test_dataset_shape_and_labels(dataset, config):
    assert dataset.X.ndim == 2
    assert dataset.X.shape[0] == dataset.y.shape[0] == dataset.groups.shape[0]
    assert dataset.X.shape[1] == len(dataset.feature_names)
    assert dataset.labels == ["bedroom", "hallway", "kitchen"]
    assert np.isfinite(dataset.X).all()


def test_dataset_groups_identify_the_source_capture(dataset):
    # One group per capture, so cross-validation cannot leak between folds.
    assert len(set(dataset.groups.tolist())) == len(dataset.captures)


def test_dataset_skips_captures_shorter_than_a_window(tmp_path, config):
    write_simulated_capture(tmp_path / "kitchen" / "long.csv", "kitchen", 150)
    write_simulated_capture(tmp_path / "bedroom" / "long.csv", "bedroom", 150)
    write_simulated_capture(tmp_path / "bedroom" / "tiny.csv", "bedroom", 5)
    dataset = build_dataset(config, data_dir=tmp_path)
    assert len(dataset.captures) == 2
    assert any("tiny" in path.name for path, _ in dataset.skipped)


def test_dataset_summary_mentions_every_room(dataset):
    summary = dataset.summary()
    for label in dataset.labels:
        assert label in summary


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "algorithm", ["random_forest", "extra_trees", "svm", "knn", "logistic"]
)
def test_every_algorithm_builds_and_fits(algorithm, dataset):
    estimator = build_estimator(
        __import__("room_detector.config", fromlist=["ModelConfig"]).ModelConfig(
            algorithm=algorithm, n_estimators=20
        )
    )
    estimator.fit(dataset.X, dataset.y)
    assert estimator.predict(dataset.X[:2]).shape == (2,)


def test_model_separates_simulated_rooms(config, dataset):
    result = RoomClassifier(config).evaluate(dataset.X, dataset.y, dataset.groups)
    # Simulated rooms are cleanly separable; this asserts the plumbing works,
    # not that real rooms would score this well.
    assert result.accuracy > 0.9
    assert result.grouped is True
    assert result.confusion.shape == (3, 3)
    assert result.confusion.sum() == dataset.X.shape[0]


def test_evaluation_report_renders(config, dataset):
    text = str(RoomClassifier(config).evaluate(dataset.X, dataset.y, dataset.groups))
    assert "accuracy" in text and "confusion matrix" in text


def test_training_needs_two_rooms(config, dataset):
    single = np.full(dataset.y.shape, "kitchen", dtype=object)
    with pytest.raises(ValueError, match="at least two rooms"):
        RoomClassifier(config).fit(dataset.X, single)


def test_predict_before_fit_is_an_error(config):
    with pytest.raises(RuntimeError, match="not trained"):
        RoomClassifier(config).predict(np.zeros((1, 5)))


def test_predict_one_returns_a_known_room(trained, dataset):
    room, confidence = trained.predict_one(dataset.X[0])
    assert room in trained.classes_
    assert 0.0 <= confidence <= 1.0


def test_wrong_feature_count_is_reported_clearly(trained):
    with pytest.raises(ValueError, match="expects"):
        trained.predict(np.zeros((1, 3)))


def test_feature_importances_are_named_and_ranked(trained):
    top = trained.feature_importances(top=5)
    assert len(top) == 5
    assert all(name in trained.feature_names for name, _ in top)
    assert [value for _, value in top] == sorted(
        (value for _, value in top), reverse=True
    )


def test_save_and_load_round_trip(trained, dataset, tmp_path):
    path = trained.save(tmp_path / "model.joblib")
    assert path.is_file()

    reloaded = RoomClassifier.load(path)
    assert reloaded.classes_ == trained.classes_
    assert reloaded.feature_names == trained.feature_names
    assert reloaded.preprocessor is not None
    np.testing.assert_array_equal(
        reloaded.predict(dataset.X[:10]), trained.predict(dataset.X[:10])
    )


def test_loading_a_missing_model_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one with"):
        RoomClassifier.load(tmp_path / "nope.joblib")


def test_loading_a_foreign_file_is_rejected(tmp_path):
    import joblib

    path = tmp_path / "junk.joblib"
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ValueError, match="not a room_detector model"):
        RoomClassifier.load(path)


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------
def test_majority_vote_prefers_the_most_frequent_room():
    room, confidence = _majority_vote([("a", 0.6), ("b", 0.9), ("a", 0.8)])
    assert room == "a"
    assert confidence == pytest.approx(0.7)


def test_majority_vote_breaks_ties_by_confidence():
    assert _majority_vote([("a", 0.4), ("b", 0.9)])[0] == "b"


def test_majority_vote_rejects_an_empty_buffer():
    with pytest.raises(ValueError):
        _majority_vote([])


def test_detector_waits_for_a_full_window(trained, config):
    from room_detector.csi import iter_csi_records

    detector = RealtimeRoomDetector(trained, config)
    records = list(iter_csi_records(simulate_lines("kitchen", 60, seed=5)))
    emitted = [detector.feed(record) for record in records[: config.features.window_size - 1]]
    assert all(prediction is None for prediction in emitted)


def test_detector_classifies_a_replayed_capture(trained, config):
    detector = RealtimeRoomDetector(trained, config)
    predictions = list(detector.process(simulate_lines("hallway", 200, seed=77)))
    assert predictions
    assert all(prediction.room == "hallway" for prediction in predictions)
    assert all(0.0 <= prediction.confidence <= 1.0 for prediction in predictions)


def test_stream_survives_boot_chatter_and_truncated_lines(trained, config):
    noise = [
        "I (531) cpu_start: Starting scheduler",
        "\x00\x01 garbage",
        "CSI_DATA,AP,3C:71:BF:6D:2A:78,-73,11,1,0,[1 2 3",
        "",
    ]
    lines = noise + simulate_lines("kitchen", 200, seed=31)
    assert run_stream(trained, lines, config, on_prediction=lambda p: None) > 0


def test_detector_requires_a_model_with_preprocessing(config, dataset):
    model = RoomClassifier(config).fit(dataset.X, dataset.y)  # no preprocessor attached
    with pytest.raises(ValueError, match="subcarrier configuration"):
        RealtimeRoomDetector(model, config)


def test_detector_reset_clears_state(trained, config):
    from room_detector.csi import iter_csi_records

    detector = RealtimeRoomDetector(trained, config)
    for record in list(iter_csi_records(simulate_lines("kitchen", 60, seed=9)))[:30]:
        detector.feed(record)
    detector.reset()
    assert detector._buffer == __import__("collections").deque(maxlen=50)
