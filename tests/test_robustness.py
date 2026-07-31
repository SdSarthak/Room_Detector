"""Regression tests for failure modes and correctness traps found in pass 2.

Every fixture here is generated deterministically -- nothing downloads, nothing
touches hardware.
"""

import numpy as np
import pytest

from room_detector import csi, preprocessing
from room_detector.cli import main
from room_detector.config import Config
from room_detector.csi import CSIParseError, iter_csi_records, parse_csi_line
from room_detector.dataset import build_dataset
from room_detector.model import RoomClassifier
from room_detector.simulate import simulate_lines, write_simulated_capture
from room_detector.stream import RealtimeRoomDetector


def _corrupt_array_token(line: str, token: str) -> str:
    """Replace the first value inside the bracketed CSI array with ``token``."""

    head, _, body = line.partition("[")
    return f"{head}[{token} {body.split(' ', 1)[1]}"


# ----------------------------------------------------------------------
# Parser: out-of-range CSI values
# ----------------------------------------------------------------------
@pytest.mark.parametrize("token", ["99999", "-129", "128", "-2147483648"])
def test_out_of_range_csi_values_are_rejected(token):
    """A value the int8 firmware buffer cannot hold means the line is corrupt.

    Before this check numpy silently wrapped 99999 to -31073, turning a garbled
    line into a plausible looking sample; under numpy >= 2 it raises
    OverflowError, which is not a ValueError and so escaped every tolerant path.
    """

    line = _corrupt_array_token(simulate_lines("kitchen", 1, seed=7)[0], token)
    with pytest.raises(CSIParseError) as excinfo:
        parse_csi_line(line)
    assert "int8" in str(excinfo.value)


def test_in_range_boundary_values_still_parse():
    line = _corrupt_array_token(simulate_lines("kitchen", 1, seed=7)[0], "-128")
    line = _corrupt_array_token(line, "127")
    assert parse_csi_line(line).csi_raw[0] == 127


def test_a_corrupt_value_does_not_abort_the_rest_of_the_capture():
    lines = simulate_lines("kitchen", 5, seed=7)
    lines[2] = _corrupt_array_token(lines[2], "99999")

    tolerant = list(iter_csi_records(lines))
    assert len(tolerant) == 4, "the four intact packets must survive one corrupt line"

    with pytest.raises(CSIParseError):
        list(iter_csi_records(lines, strict=True))


# ----------------------------------------------------------------------
# Hampel filter: chunking must not change the answer
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_packets", [3, 9, 10, 11, 40])
@pytest.mark.parametrize("window", [1, 3, 5, 9])
def test_chunked_hampel_matches_the_unchunked_result(monkeypatch, n_packets, window):
    """Chunk boundaries must be invisible: chunk 4 == chunk 1_000_000."""

    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(n_packets, 4))
    matrix[rng.integers(0, n_packets, size=max(1, n_packets // 4)), 0] += 40.0

    monkeypatch.setattr(preprocessing, "_HAMPEL_CHUNK", 1_000_000)
    whole = preprocessing.hampel_filter(matrix, window)
    monkeypatch.setattr(preprocessing, "_HAMPEL_CHUNK", 4)
    chunked = preprocessing.hampel_filter(matrix, window)

    assert np.array_equal(whole, chunked)


def test_hampel_still_removes_spikes_across_a_chunk_boundary(monkeypatch):
    monkeypatch.setattr(preprocessing, "_HAMPEL_CHUNK", 4)
    matrix = np.ones((20, 2))
    matrix[4, 0] = 500.0   # first row of the second chunk
    matrix[7, 1] = -500.0  # last row of the second chunk
    cleaned = preprocessing.hampel_filter(matrix, 5)
    assert cleaned[4, 0] == pytest.approx(1.0)
    assert cleaned[7, 1] == pytest.approx(1.0)


def test_hampel_does_not_mutate_its_input():
    matrix = np.ones((12, 2))
    matrix[6, 0] = 99.0
    before = matrix.copy()
    preprocessing.hampel_filter(matrix, 5)
    assert np.array_equal(matrix, before)


# ----------------------------------------------------------------------
# Grouped cross-validation must refuse a dataset it cannot score
# ----------------------------------------------------------------------
def test_grouped_cv_refuses_one_capture_per_room(tmp_path):
    """With one capture per room every fold trains without the room it tests.

    That silently scored 0.0 accuracy on perfectly separable simulated data,
    which reads as "the model is broken" rather than "the dataset is too small".
    """

    for room in ("kitchen", "bedroom"):
        write_simulated_capture(tmp_path / room / "capture_00.csv", room, 160)
    config = Config(data_dir=tmp_path)
    config.features.window_size = 50
    config.features.window_step = 25
    config.model.n_estimators = 20
    dataset = build_dataset(config)

    assert len(set(dataset.groups.tolist())) == 2
    with pytest.raises(ValueError) as excinfo:
        RoomClassifier(config).evaluate(dataset.X, dataset.y, dataset.groups)
    message = str(excinfo.value)
    assert "at least two captures per room" in message
    assert "bedroom" in message and "kitchen" in message


def test_grouped_cv_scores_two_captures_per_room(tmp_path):
    for room in ("kitchen", "bedroom"):
        for index in range(2):
            write_simulated_capture(
                tmp_path / room / f"capture_{index:02d}.csv", room, 160, seed=index + 3
            )
    config = Config(data_dir=tmp_path)
    config.features.window_size = 50
    config.features.window_step = 25
    config.model.n_estimators = 20
    dataset = build_dataset(config)

    result = RoomClassifier(config).evaluate(dataset.X, dataset.y, dataset.groups)
    assert result.grouped and result.n_folds >= 2
    assert result.accuracy > 0.5, "separable simulated rooms must beat chance"


def test_train_command_survives_a_dataset_grouped_cv_cannot_score(tmp_path, capsys):
    """`train` must still produce a model, just without a cross-validation score."""

    for room in ("kitchen", "bedroom"):
        write_simulated_capture(tmp_path / "data" / room / "c0.csv", room, 160)
    model_path = tmp_path / "model.joblib"
    exit_code = main([
        "--data-dir", str(tmp_path / "data"), "--model-path", str(model_path),
        "train", "--window-size", "50", "--window-step", "25",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert model_path.is_file()
    assert "skipping cross-validation" in out
    assert "at least two captures per room" in out


# ----------------------------------------------------------------------
# Live prediction must use the settings the model was trained with
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    """A model trained with window_size=50, plus a replayable capture."""

    root = tmp_path_factory.mktemp("bound")
    for room in ("kitchen", "bedroom"):
        for index in range(2):
            write_simulated_capture(root / room / f"c{index}.csv", room, 160, seed=index + 5)
    config = Config(data_dir=root)
    config.features.window_size = 50
    config.features.window_step = 25
    config.model.n_estimators = 40
    model = RoomClassifier(config).fit_dataset(build_dataset(config))
    path = model.save(root / "model.joblib")
    return RoomClassifier.load(path), root / "kitchen" / "c0.csv"


def test_detector_ignores_a_window_size_the_model_was_not_trained_on(small_model, capsys):
    """The feature vector is the same length for every window size.

    Per-subcarrier statistics do not depend on how many packets they summarise,
    so a mismatched window_size raises nothing and simply returns confident
    nonsense. The detector must pin the training values instead.
    """

    model, _ = small_model
    wrong = Config()
    wrong.features.window_size = 8
    wrong.features.window_step = 4

    detector = RealtimeRoomDetector(model, wrong)
    assert detector.config.features.window_size == 50
    assert detector.config.features.window_step == 25
    assert detector.config.preprocess == model.config.preprocess
    assert "trained with" in capsys.readouterr().err


def test_detector_keeps_the_callers_stream_settings(small_model, capsys):
    model, _ = small_model
    caller = Config()
    caller.features.window_size = model.config.features.window_size
    caller.features.window_step = model.config.features.window_step
    caller.preprocess = model.config.preprocess
    caller.stream.min_confidence = 0.75
    caller.stream.vote_window = 7

    detector = RealtimeRoomDetector(model, caller)
    assert detector.config.stream.min_confidence == 0.75
    assert detector.config.stream.vote_window == 7
    assert capsys.readouterr().err == "", "identical settings must not warn"


def test_detector_counts_classified_windows_even_when_all_are_suppressed(small_model):
    model, capture = small_model
    config = Config()
    config.stream.min_confidence = 1.0  # nothing can clear this
    detector = RealtimeRoomDetector(model, config)

    lines = capture.read_text(encoding="utf-8").splitlines()
    assert list(detector.process(lines)) == []
    assert detector.windows_classified > 0, "windows were classified, just suppressed"


def test_predict_distinguishes_suppression_from_a_short_capture(small_model, capsys):
    model, capture = small_model
    exit_code = main([
        "--model-path", str(capture.parent.parent / "model.joblib"),
        "predict", "--file", str(capture), "--min-confidence", "1.0",
    ])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "minimum confidence" in err
    assert "fewer than" not in err


# ----------------------------------------------------------------------
# collect: bounded recording must actually be bounded
# ----------------------------------------------------------------------
def test_collect_honours_seconds_on_a_stream_with_no_csi(tmp_path, monkeypatch, capsys):
    """A board that only emits boot chatter used to hang `--seconds` forever.

    The deadline was checked after the CSI_MARKER filter, so a stream that never
    produced a CSI line never reached the check.
    """

    def chatter():
        while True:
            yield "I (245) wifi:mode : sta"

    monkeypatch.setattr("room_detector.cli.stdin_lines", chatter)
    exit_code = main([
        "--data-dir", str(tmp_path), "collect",
        "--room", "kitchen", "--stdin", "--seconds", "0.2",
    ])
    assert exit_code == 1  # bounded, and reports that nothing was captured
    assert "no CSI packets captured" in capsys.readouterr().err


def test_collect_honours_seconds_on_an_idle_stream(tmp_path, monkeypatch, capsys):
    """A silent serial port yields empty strings on timeout, not nothing."""

    def idle():
        while True:
            yield ""

    monkeypatch.setattr("room_detector.cli.stdin_lines", idle)
    assert main([
        "--data-dir", str(tmp_path), "collect",
        "--room", "kitchen", "--stdin", "--seconds", "0.2",
    ]) == 1
    assert "no CSI packets captured" in capsys.readouterr().err


def test_collect_stops_at_the_packet_limit(tmp_path, monkeypatch):
    lines = simulate_lines("kitchen", 40, seed=2)

    monkeypatch.setattr("room_detector.cli.stdin_lines", lambda: iter(lines))
    assert main([
        "--data-dir", str(tmp_path), "collect",
        "--room", "kitchen", "--stdin", "--packets", "12",
    ]) == 0
    written = (tmp_path / "kitchen").glob("capture_*.csv")
    recorded = next(written).read_text(encoding="utf-8").strip().splitlines()
    assert len(recorded) == 12


@pytest.mark.parametrize(
    "flags, expected",
    [(["--packets", "0"], "--packets"), (["--seconds", "0"], "--seconds"),
     (["--packets", "-5"], "--packets"), (["--seconds", "-1"], "--seconds")],
)
def test_collect_rejects_meaningless_limits(tmp_path, capsys, flags, expected):
    """0 was falsy and silently meant "no limit"; negatives were accepted too."""

    assert main(["--data-dir", str(tmp_path), "collect",
                 "--room", "kitchen", "--stdin", *flags]) == 2
    assert expected in capsys.readouterr().err


# ----------------------------------------------------------------------
# Serial source
# ----------------------------------------------------------------------
def test_serial_read_timeout_yields_control_back(monkeypatch):
    """`serial_lines` must not spin silently while the board is quiet."""

    serial = pytest.importorskip("serial")

    class FakePort:
        def __init__(self, *args, **kwargs):
            self.reads = [b"", b"", b"CSI_DATA,AP\n"]

        def readline(self):
            return self.reads.pop(0) if self.reads else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True
            return False

    monkeypatch.setattr(serial, "Serial", FakePort)
    from room_detector.stream import serial_lines

    stream = serial_lines("COM_FAKE")
    assert [next(stream) for _ in range(3)] == ["", "", "CSI_DATA,AP"]
    stream.close()


def test_empty_lines_are_ignored_by_the_record_iterator():
    lines = ["", "", *simulate_lines("kitchen", 3, seed=4), ""]
    assert len(list(iter_csi_records(lines))) == 3
    assert len(list(iter_csi_records(lines, strict=True))) == 3


# ----------------------------------------------------------------------
# Parser invariants that the marker constant depends on
# ----------------------------------------------------------------------
def test_marker_only_lines_are_not_packets():
    assert list(iter_csi_records([csi.CSI_MARKER])) == []
