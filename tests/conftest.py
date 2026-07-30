import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from room_detector.simulate import simulate_dataset, simulate_lines  # noqa: E402

#: A capture recorded from real hardware, shipped with the ESP32-CSI-Tool.
VENDOR_CAPTURE = REPO_ROOT / "ESP32-CSI-Tool" / "python_utils" / "example_csi.csv"


@pytest.fixture(scope="session")
def vendor_capture() -> Path:
    """Path to the upstream example capture, skipping if the submodule is absent."""

    if not VENDOR_CAPTURE.is_file():
        pytest.skip(
            "ESP32-CSI-Tool submodule not checked out; "
            "run `git submodule update --init --recursive`"
        )
    return VENDOR_CAPTURE


@pytest.fixture(scope="session")
def sample_lines() -> list[str]:
    """A deterministic simulated capture of a single room."""

    return simulate_lines("kitchen", 80, seed=1234)


@pytest.fixture(scope="session")
def simulated_data_dir(tmp_path_factory) -> Path:
    """A small three-room dataset on disk, generated once per test session."""

    root = tmp_path_factory.mktemp("csi_data")
    simulate_dataset(root, captures_per_room=3, packets_per_capture=150)
    return root
