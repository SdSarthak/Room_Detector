"""Tests for configuration handling and the command line interface."""

import os
from pathlib import Path

import pytest

from room_detector.cli import build_parser, main
from room_detector.config import (
    Config,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    StreamConfig,
    load_dotenv,
)
from room_detector.simulate import write_simulated_capture


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------
def test_defaults_are_coherent():
    config = Config()
    assert config.features.window_step <= config.features.window_size
    assert isinstance(config.data_dir, Path)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PreprocessConfig(normalize="nope"),
        lambda: PreprocessConfig(hampel_window=-1),
        lambda: FeatureConfig(window_size=1),
        lambda: FeatureConfig(window_step=0),
        lambda: FeatureConfig(use_amplitude=False, use_phase=False, use_rssi=False,
                              use_motion=False),
        lambda: ModelConfig(algorithm="deep_magic"),
        lambda: ModelConfig(cv_folds=1),
        lambda: StreamConfig(vote_window=0),
        lambda: StreamConfig(min_confidence=1.5),
    ],
)
def test_invalid_settings_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_string_paths_become_path_objects():
    config = Config(data_dir="some/dir", model_path="some/model.joblib")
    assert isinstance(config.data_dir, Path)
    assert isinstance(config.model_path, Path)


def test_to_dict_and_from_dict_round_trip():
    config = Config()
    config.features.window_size = 32
    config.model.algorithm = "svm"
    restored = Config.from_dict(config.to_dict())
    assert restored.features.window_size == 32
    assert restored.model.algorithm == "svm"
    assert restored.data_dir == config.data_dir


def test_from_dict_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown model option"):
        Config.from_dict({"model": {"learning_rate": 0.1}})


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("ROOM_DETECTOR_WINDOW_SIZE", "16")
    monkeypatch.setenv("ROOM_DETECTOR_ALGORITHM", "knn")
    monkeypatch.setenv("ROOM_DETECTOR_DATA_DIR", "captures")
    config = Config.from_env()
    assert config.features.window_size == 16
    assert config.model.algorithm == "knn"
    assert config.data_dir == Path("captures")


def test_from_env_reports_a_bad_number(monkeypatch):
    monkeypatch.setenv("ROOM_DETECTOR_WINDOW_SIZE", "sixteen")
    with pytest.raises(ValueError, match="not an integer"):
        Config.from_env()


def test_load_dotenv_does_not_override_the_real_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\nROOM_DETECTOR_ALGORITHM=svm\nROOM_DETECTOR_WINDOW_SIZE=8\n\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ROOM_DETECTOR_ALGORITHM", raising=False)
    monkeypatch.setenv("ROOM_DETECTOR_WINDOW_SIZE", "99")
    assert load_dotenv(env_file) is True
    assert os.environ["ROOM_DETECTOR_ALGORITHM"] == "svm"
    assert os.environ["ROOM_DETECTOR_WINDOW_SIZE"] == "99"
    monkeypatch.delenv("ROOM_DETECTOR_ALGORITHM", raising=False)


def test_load_dotenv_tolerates_a_missing_file(tmp_path):
    assert load_dotenv(tmp_path / "absent") is False


def test_from_yaml_reports_a_missing_file(tmp_path):
    pytest.importorskip("yaml")
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "absent.yaml")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("command", ["collect", "inspect", "train", "evaluate", "predict"])
def test_every_command_is_registered(command):
    assert command in build_parser().format_help()


def test_inspect_summarises_a_capture(tmp_path, capsys):
    path = write_simulated_capture(tmp_path / "kitchen.csv", "kitchen", 120)
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "packets:          120" in out
    assert "subcarriers:      64" in out
    assert "training windows" in out


def test_inspect_reports_a_missing_file(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.csv")]) == 1
    assert "not found" in capsys.readouterr().err


def test_train_then_predict_over_the_cli(tmp_path, capsys):
    data_dir = tmp_path / "data"
    for room in ("kitchen", "bedroom"):
        for index in range(2):
            write_simulated_capture(data_dir / room / f"c{index}.csv", room, 150,
                                    seed=hash((room, index)) % 10_000)
    model_path = tmp_path / "model.joblib"

    assert main([
        "--data-dir", str(data_dir), "--model-path", str(model_path),
        "train", "--window-size", "50", "--window-step", "25",
    ]) == 0
    assert model_path.is_file()
    assert "saved model" in capsys.readouterr().out

    assert main([
        "--model-path", str(model_path),
        "predict", "--file", str(data_dir / "kitchen" / "c0.csv"),
    ]) == 0
    assert "room=kitchen" in capsys.readouterr().out


def test_train_reports_an_empty_data_directory(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path / "empty"), "train"]) == 1
    assert "error:" in capsys.readouterr().err


def test_predict_reports_a_missing_model(tmp_path, capsys):
    path = write_simulated_capture(tmp_path / "k.csv", "kitchen", 80)
    assert main(["--model-path", str(tmp_path / "absent.joblib"),
                 "predict", "--file", str(path)]) == 1
    assert "Train one with" in capsys.readouterr().err


def test_cli_window_overrides_are_validated(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "train", "--window-size", "1"]) == 1
    assert "window_size" in capsys.readouterr().err


def test_predict_reports_a_capture_shorter_than_one_window(tmp_path, capsys):
    data_dir = tmp_path / "data"
    for room in ("kitchen", "bedroom"):
        for index in range(2):
            write_simulated_capture(data_dir / room / f"c{index}.csv", room, 150,
                                    seed=index + 1)
    model_path = tmp_path / "model.joblib"
    main(["--data-dir", str(data_dir), "--model-path", str(model_path),
          "train", "--window-size", "50", "--window-step", "25"])
    capsys.readouterr()

    short = write_simulated_capture(tmp_path / "short.csv", "kitchen", 10)
    assert main(["--model-path", str(model_path), "predict", "--file", str(short)]) == 1
    assert "fewer than" in capsys.readouterr().err
