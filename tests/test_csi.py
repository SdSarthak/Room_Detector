"""Tests for the ESP32-CSI-Tool line parser."""

import math

import numpy as np
import pytest

from room_detector.csi import (
    CSI_COLUMNS,
    CSIParseError,
    iter_csi_records,
    parse_csi_line,
    raw_to_complex,
    read_csi_file,
    records_to_arrays,
)

GOOD_LINE = (
    "CSI_DATA,AP,3C:71:BF:6D:2A:78,-73,11,1,0,1,1,1,0,0,0,0,-93,0,1,1,"
    "80272146,0,101,0,0,80.363225,384,[1 2 3 4 -5 -6 ]"
)


def test_parses_every_scalar_field():
    record = parse_csi_line(GOOD_LINE)
    assert record.role == "AP"
    assert record.mac == "3C:71:BF:6D:2A:78"
    assert record.rssi == -73
    assert record.rate == 11
    assert record.noise_floor == -93
    assert record.channel == 1
    assert record.local_timestamp == 80272146
    assert record.sig_len == 101
    assert record.real_timestamp == pytest.approx(80.363225)
    assert record.declared_len == 384


def test_column_list_matches_dataclass_fields():
    record = parse_csi_line(GOOD_LINE)
    for column in CSI_COLUMNS:
        assert hasattr(record, column)


def test_interleaving_is_imaginary_then_real():
    # Buffer is [imag0, real0, imag1, real1, ...] per csi_component.h.
    record = parse_csi_line(GOOD_LINE)
    assert record.n_subcarriers == 3
    np.testing.assert_array_equal(record.imaginary, [1, 3, -5])
    np.testing.assert_array_equal(record.real, [2, 4, -6])


def test_amplitude_and_phase_match_the_vendor_reference():
    # ESP32-CSI-Tool/python_utils/parse_csi.py computes exactly this.
    record = parse_csi_line(GOOD_LINE)
    expected_amp = [math.sqrt(1**2 + 2**2), math.sqrt(3**2 + 4**2), math.sqrt(5**2 + 6**2)]
    expected_phase = [math.atan2(1, 2), math.atan2(3, 4), math.atan2(-5, -6)]
    np.testing.assert_allclose(record.amplitude(), expected_amp)
    np.testing.assert_allclose(record.phase(), expected_phase)


def test_complex_uses_real_plus_j_imag():
    record = parse_csi_line(GOOD_LINE)
    np.testing.assert_allclose(record.complex(), [2 + 1j, 4 + 3j, -6 - 5j])
    np.testing.assert_allclose(np.abs(record.complex()), record.amplitude())


def test_declared_len_is_ignored_in_favour_of_the_actual_array():
    # The firmware reports 384 while emitting 6 values when built LLTF-only.
    record = parse_csi_line(GOOD_LINE)
    assert record.declared_len == 384
    assert record.n_subcarriers == 3


def test_leading_serial_noise_is_stripped():
    record = parse_csi_line("I (5533) wifi: mode set\x00" + GOOD_LINE)
    assert record.rssi == -73


@pytest.mark.parametrize(
    "line",
    [
        "",
        "I (123) boot: nothing to see here",
        "CSI_DATA,AP,3C:71:BF:6D:2A:78,-73,[1 2",          # truncated mid-array
        "CSI_DATA,AP,MAC,-73,11,[1 2 3 4]",                 # too few scalar fields
        GOOD_LINE.replace("[1 2 3 4 -5 -6 ]", "[1 2 3]"),   # odd value count
        GOOD_LINE.replace("[1 2 3 4 -5 -6 ]", "[]"),        # empty array
        GOOD_LINE.replace("-73,11", "abc,11"),              # non-integer rssi
        GOOD_LINE.replace("[1 2 3 4 -5 -6 ]", "[1 x 3 4]"), # non-integer sample
    ],
)
def test_malformed_lines_raise(line):
    with pytest.raises(CSIParseError):
        parse_csi_line(line)


def test_expected_subcarriers_is_enforced():
    with pytest.raises(CSIParseError, match="expected 64 subcarriers"):
        parse_csi_line(GOOD_LINE, expected_subcarriers=64)


#: The header ESP32-CSI-Tool writes. Its last column is literally named
#: CSI_DATA, so it matches the packet marker without being a packet.
FIRMWARE_HEADER = (
    "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
    "aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,"
    "secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,"
    "real_timestamp,len,CSI_DATA"
)


def test_csv_header_is_skipped_even_in_strict_mode():
    records = list(iter_csi_records([FIRMWARE_HEADER, GOOD_LINE], strict=True))
    assert len(records) == 1


def test_header_written_capture_reads_back(tmp_path):
    path = tmp_path / "capture.csv"
    path.write_text(f"{FIRMWARE_HEADER}\n{GOOD_LINE}\n{GOOD_LINE}\n", encoding="utf-8")
    assert len(read_csi_file(path, strict=True)) == 2


def test_iter_skips_bad_lines_but_strict_raises():
    lines = ["boot chatter", GOOD_LINE, "CSI_DATA,AP,broken,[1 2", GOOD_LINE]
    assert len(list(iter_csi_records(lines))) == 2
    with pytest.raises(CSIParseError):
        list(iter_csi_records(lines, strict=True))


def test_raw_to_complex_rejects_odd_length():
    with pytest.raises(ValueError):
        raw_to_complex([1, 2, 3])


def test_records_to_arrays_drops_minority_subcarrier_widths():
    wide = GOOD_LINE
    narrow = GOOD_LINE.replace("[1 2 3 4 -5 -6 ]", "[1 2 ]")
    records = [parse_csi_line(line) for line in (wide, wide, wide, narrow)]
    arrays = records_to_arrays(records)
    assert arrays["amplitude"].shape == (3, 3)
    assert int(arrays["n_dropped"]) == 1


def test_records_to_arrays_needs_records():
    with pytest.raises(ValueError):
        records_to_arrays([])


# ----------------------------------------------------------------------
# Against the real hardware capture shipped with ESP32-CSI-Tool
# ----------------------------------------------------------------------
def test_reads_the_vendor_example_capture(vendor_capture):
    records = read_csi_file(vendor_capture, strict=True)
    assert len(records) == 13
    assert all(record.n_subcarriers == 64 for record in records)
    assert records[0].mac == "3C:71:BF:6D:2A:78"
    assert records[0].rssi == -73
    # The declared length swings between 128 and 384 across this very capture
    # while every packet actually carries 64 subcarriers -- which is exactly why
    # the parser sizes packets from the array instead of the `len` field.
    assert {record.declared_len for record in records} == {128, 384}
    assert {record.n_subcarriers for record in records} == {64}


def test_vendor_capture_guard_subcarriers_are_null(vendor_capture):
    arrays = records_to_arrays(read_csi_file(vendor_capture))
    null = np.flatnonzero(np.all(arrays["amplitude"] == 0, axis=0))
    # Guard bands at the edges of the 64-bin LLTF symbol.
    assert set(null.tolist()) == {2, 3, 4, 60, 61, 62, 63}
