"""Tests for windowing and fingerprint feature extraction."""

import numpy as np
import pytest

from room_detector.config import FeatureConfig
from room_detector.features import (
    AMPLITUDE_STATS,
    PHASE_STATS,
    extract_features,
    feature_names,
    sliding_windows,
    window_feature_vector,
)

N_SC = 8
WINDOW = 10


def _window(seed=0, n=WINDOW, n_sc=N_SC):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 20, size=(n, n_sc))


def test_sliding_windows_covers_the_capture():
    assert sliding_windows(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_sliding_windows_is_empty_for_short_captures():
    assert sliding_windows(3, 4, 2) == []


def test_sliding_windows_step_equal_to_size_does_not_overlap():
    bounds = sliding_windows(12, 4, 4)
    assert bounds == [(0, 4), (4, 8), (8, 12)]


@pytest.mark.parametrize("size,step", [(0, 1), (4, 0)])
def test_sliding_windows_validates_arguments(size, step):
    with pytest.raises(ValueError):
        sliding_windows(10, size, step)


def test_feature_vector_length_matches_the_name_list():
    config = FeatureConfig(window_size=WINDOW)
    vector = window_feature_vector(
        _window(), _window(1), np.full(WINDOW, -60.0), np.full(WINDOW, -92.0), config
    )
    names = feature_names(N_SC, config)
    assert vector.shape == (len(names),)


def test_feature_vector_composition_is_what_it_claims():
    config = FeatureConfig(window_size=WINDOW)
    expected = (
        len(AMPLITUDE_STATS) * N_SC  # amplitude stats
        + N_SC                       # motion
        + len(PHASE_STATS) * N_SC    # phase stats
        + 6                          # rssi / noise scalars
    )
    vector = window_feature_vector(
        _window(), _window(1), np.full(WINDOW, -60.0), np.full(WINDOW, -92.0), config
    )
    assert vector.size == expected


@pytest.mark.parametrize(
    "config",
    [
        FeatureConfig(window_size=WINDOW, use_phase=False),
        FeatureConfig(window_size=WINDOW, use_rssi=False),
        FeatureConfig(window_size=WINDOW, use_motion=False),
        FeatureConfig(window_size=WINDOW, use_amplitude=False, use_phase=False),
    ],
)
def test_toggling_blocks_keeps_names_and_values_aligned(config):
    vector = window_feature_vector(
        _window(),
        _window(1) if config.use_phase else None,
        np.full(WINDOW, -60.0) if config.use_rssi else None,
        np.full(WINDOW, -92.0) if config.use_rssi else None,
        config,
    )
    assert vector.size == len(feature_names(N_SC, config))


def test_amplitude_stats_are_the_real_statistics():
    config = FeatureConfig(window_size=WINDOW, use_phase=False, use_rssi=False, use_motion=False)
    amplitude = _window()
    vector = window_feature_vector(amplitude, config=config)
    np.testing.assert_allclose(vector[:N_SC], amplitude.mean(axis=0))
    np.testing.assert_allclose(vector[N_SC:2 * N_SC], amplitude.std(axis=0))
    np.testing.assert_allclose(vector[2 * N_SC:3 * N_SC], np.median(amplitude, axis=0))


def test_motion_feature_is_zero_for_a_static_channel():
    config = FeatureConfig(window_size=WINDOW, use_amplitude=False, use_phase=False,
                           use_rssi=False, use_motion=True)
    static = np.ones((WINDOW, N_SC)) * 12.0
    np.testing.assert_allclose(window_feature_vector(static, config=config), np.zeros(N_SC))


def test_motion_feature_grows_with_fluctuation():
    config = FeatureConfig(window_size=WINDOW, use_amplitude=False, use_phase=False,
                           use_rssi=False, use_motion=True)
    calm = window_feature_vector(_window(seed=2) * 0.01, config=config)
    busy = window_feature_vector(_window(seed=2), config=config)
    assert busy.sum() > calm.sum()


def test_snr_feature_is_rssi_minus_noise_floor():
    config = FeatureConfig(window_size=WINDOW, use_amplitude=False, use_phase=False,
                           use_motion=False, use_rssi=True)
    vector = window_feature_vector(
        _window(), None, np.full(WINDOW, -60.0), np.full(WINDOW, -92.0), config
    )
    # [mean, std, min, max, noise_mean, snr]
    assert vector[0] == pytest.approx(-60.0)
    assert vector[4] == pytest.approx(-92.0)
    assert vector[5] == pytest.approx(32.0)


def test_features_are_deterministic():
    config = FeatureConfig(window_size=WINDOW)
    args = (_window(), _window(1), np.full(WINDOW, -60.0), np.full(WINDOW, -92.0), config)
    np.testing.assert_array_equal(window_feature_vector(*args), window_feature_vector(*args))


def test_missing_phase_or_rssi_is_an_explicit_error():
    config = FeatureConfig(window_size=WINDOW)
    with pytest.raises(ValueError, match="use_phase"):
        window_feature_vector(_window(), None, np.full(WINDOW, -60.0), None, config)
    with pytest.raises(ValueError, match="use_rssi"):
        window_feature_vector(_window(), _window(1), None, None,
                              FeatureConfig(window_size=WINDOW, use_phase=True))


def test_mismatched_window_lengths_are_rejected():
    config = FeatureConfig(window_size=WINDOW, use_phase=False)
    with pytest.raises(ValueError, match="rssi window"):
        window_feature_vector(_window(), None, np.full(3, -60.0), None, config)


def test_extract_features_produces_one_row_per_window():
    config = FeatureConfig(window_size=4, window_step=2, use_phase=False, use_rssi=False)
    amplitude = _window(n=10)
    features = extract_features(amplitude, config=config)
    assert features.shape[0] == len(sliding_windows(10, 4, 2))


def test_extract_features_refuses_a_capture_shorter_than_one_window():
    config = FeatureConfig(window_size=64, use_phase=False, use_rssi=False)
    with pytest.raises(ValueError, match="fewer than the"):
        extract_features(_window(n=10), config=config)


def test_feature_names_are_unique():
    names = feature_names(N_SC, FeatureConfig(window_size=WINDOW))
    assert len(names) == len(set(names))
