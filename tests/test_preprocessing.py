"""Tests for CSI cleaning."""

import numpy as np
import pytest

from room_detector.config import PreprocessConfig
from room_detector.preprocessing import (
    LLTF_DATA_SUBCARRIERS,
    LLTF_SUBCARRIERS,
    CSIPreprocessor,
    hampel_filter,
    moving_average,
    normalize,
    null_subcarrier_mask,
    sanitize_phase,
    select_subcarriers,
)


def test_data_subcarrier_set_matches_80211_20mhz():
    # 52 data subcarriers, skipping guard bands and the DC bin.
    assert len(LLTF_DATA_SUBCARRIERS) == 52
    assert 32 not in LLTF_DATA_SUBCARRIERS  # DC
    assert 0 not in LLTF_DATA_SUBCARRIERS and 63 not in LLTF_DATA_SUBCARRIERS


def test_select_subcarriers_defaults_to_the_data_bins():
    matrix = np.arange(3 * LLTF_SUBCARRIERS, dtype=float).reshape(3, LLTF_SUBCARRIERS)
    selected = select_subcarriers(matrix)
    assert selected.shape == (3, 52)
    np.testing.assert_array_equal(selected[0], matrix[0, list(LLTF_DATA_SUBCARRIERS)])


def test_select_subcarriers_passes_through_non_lltf_widths():
    matrix = np.ones((3, 100))
    assert select_subcarriers(matrix).shape == (3, 100)


def test_select_subcarriers_rejects_out_of_range_index():
    with pytest.raises(IndexError):
        select_subcarriers(np.ones((2, 4)), indices=[0, 9])


def test_null_subcarrier_mask_finds_constant_zero_columns():
    matrix = np.array([[1.0, 0.0, 3.0], [2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(null_subcarrier_mask(matrix), [False, True, False])


def test_hampel_replaces_a_spike_with_the_local_median():
    clean = np.full((21, 1), 10.0)
    spiked = clean.copy()
    spiked[10, 0] = 500.0
    filtered = hampel_filter(spiked, window=5, n_sigmas=3.0)
    assert filtered[10, 0] == pytest.approx(10.0)
    assert filtered.shape == spiked.shape


def test_hampel_mostly_leaves_clean_data_alone():
    # A short MAD window is a noisy scale estimate, so a few clean samples are
    # always rewritten. What matters is that the filter is not destructive.
    rng = np.random.default_rng(0)
    data = rng.normal(50, 1.0, size=(60, 4))
    filtered = hampel_filter(data, window=5, n_sigmas=6.0)
    unchanged = np.isclose(filtered, data).mean()
    assert unchanged > 0.9
    assert abs(filtered.mean() - data.mean()) < 0.1


def test_hampel_is_a_noop_for_tiny_captures():
    data = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(hampel_filter(data), data)


def test_moving_average_preserves_shape_and_smooths():
    data = np.array([[0.0], [10.0], [0.0], [10.0], [0.0]])
    smoothed = moving_average(data, window=3)
    assert smoothed.shape == data.shape
    assert smoothed.std() < data.std()


def test_moving_average_window_1_is_identity():
    data = np.arange(10.0).reshape(5, 2)
    np.testing.assert_allclose(moving_average(data, window=1), data)


def test_sanitize_phase_removes_a_linear_ramp():
    # A pure sampling-time-offset ramp carries no room information; it must go.
    k = np.arange(52)
    ramp = np.vstack([0.05 * k + 0.7, 0.11 * k - 0.3])
    cleaned = sanitize_phase(ramp)
    np.testing.assert_allclose(cleaned, np.zeros_like(ramp), atol=1e-9)


def test_sanitize_phase_keeps_non_linear_structure():
    k = np.arange(52)
    structure = np.sin(2 * np.pi * k / 13.0)
    observed = (0.2 * k + 1.0 + structure).reshape(1, -1)
    cleaned = sanitize_phase(observed)[0]
    # The residual should still correlate strongly with the real structure.
    assert np.corrcoef(cleaned, structure - structure.mean())[0, 1] > 0.95


def test_sanitize_phase_is_invariant_to_the_offset_it_removes():
    k = np.arange(52)
    base = np.sin(2 * np.pi * k / 9.0).reshape(1, -1)
    shifted = base + 0.3 * k + 2.0
    np.testing.assert_allclose(sanitize_phase(base), sanitize_phase(shifted), atol=1e-9)


@pytest.mark.parametrize("method", ["l2", "zscore", "minmax", "none"])
def test_normalize_shapes_and_ranges(method):
    rng = np.random.default_rng(1)
    data = rng.uniform(1, 100, size=(5, 8))
    out = normalize(data, method)
    assert out.shape == data.shape
    if method == "l2":
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)
    elif method == "zscore":
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    elif method == "minmax":
        np.testing.assert_allclose(out.min(axis=1), 0.0)
        np.testing.assert_allclose(out.max(axis=1), 1.0)
    else:
        np.testing.assert_allclose(out, data)


def test_normalize_survives_an_all_zero_row():
    out = normalize(np.zeros((2, 4)), "l2")
    assert np.isfinite(out).all()


def test_normalize_rejects_unknown_method():
    with pytest.raises(ValueError):
        normalize(np.ones((2, 2)), "bogus")


# ----------------------------------------------------------------------
# CSIPreprocessor
# ----------------------------------------------------------------------
#: A dead bin that falls *inside* the 52 data subcarriers, unlike the guard bins.
DEAD_DATA_BIN = 10


def _lltf_matrix(n=30):
    rng = np.random.default_rng(7)
    matrix = rng.uniform(5, 50, size=(n, LLTF_SUBCARRIERS))
    matrix[:, [0, 1, 2, 62, 63]] = 0.0  # guard bins, already outside the data set
    matrix[:, DEAD_DATA_BIN] = 0.0
    return matrix


def test_preprocessor_drops_guard_subcarriers():
    # Guard bins are excluded by the LLTF layout, before any null detection.
    pre = CSIPreprocessor(PreprocessConfig(drop_null_subcarriers=False)).fit(_lltf_matrix())
    assert pre.n_subcarriers == 52
    assert 0 not in pre.subcarrier_indices.tolist()
    assert 32 not in pre.subcarrier_indices.tolist()


def test_preprocessor_also_drops_dead_bins_inside_the_data_set():
    pre = CSIPreprocessor(PreprocessConfig()).fit(_lltf_matrix())
    assert pre.n_subcarriers == 51
    assert DEAD_DATA_BIN not in pre.subcarrier_indices.tolist()


def test_preprocessor_transform_requires_fit():
    with pytest.raises(RuntimeError):
        CSIPreprocessor().transform(_lltf_matrix())


def test_preprocessor_layout_is_stable_across_calls():
    pre = CSIPreprocessor().fit(_lltf_matrix())
    first, _ = pre.transform(_lltf_matrix(30))
    second, _ = pre.transform(_lltf_matrix(12))
    assert first.shape[1] == second.shape[1] == pre.n_subcarriers


def test_preprocessor_rejects_a_narrower_capture_than_it_was_fitted_on():
    pre = CSIPreprocessor().fit(_lltf_matrix())
    with pytest.raises(ValueError, match="subcarriers"):
        pre.transform(np.ones((5, 10)))


def test_preprocessor_rejects_mismatched_phase_shape():
    pre = CSIPreprocessor().fit(_lltf_matrix())
    with pytest.raises(ValueError, match="does not match"):
        pre.transform(_lltf_matrix(30), np.ones((30, 10)))


def test_preprocessor_never_drops_every_subcarrier():
    pre = CSIPreprocessor().fit(np.zeros((5, LLTF_SUBCARRIERS)))
    assert pre.n_subcarriers > 0


def test_preprocessor_returns_none_phase_when_none_given():
    amp, phase = CSIPreprocessor().fit_transform(_lltf_matrix())
    assert phase is None
    assert amp.shape[0] == 30


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((0, 4)), np.ones((4, 0))])
def test_rejects_non_2d_or_empty(bad):
    with pytest.raises(ValueError):
        null_subcarrier_mask(bad)
