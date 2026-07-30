"""Synthetic CSI captures, for testing and for trying the pipeline without hardware.

Each simulated room gets its own multipath profile: a handful of reflected
paths with fixed delays and gains, which produce a frequency-selective channel
response that is stable for that room and different from every other room.
That is exactly the structure real CSI fingerprinting exploits, so a model
trained on simulated captures exercises the same code path as one trained on
real ones -- it just cannot tell you anything about real-world accuracy.

The emitted lines are byte-for-byte in the format of
``ESP32-CSI-Tool/_components/csi_component.h``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocessing import LLTF_DATA_SUBCARRIERS, LLTF_SUBCARRIERS

__all__ = ["room_channel", "simulate_lines", "write_simulated_capture", "simulate_dataset"]

#: Full-scale value of the int8 CSI buffer the ESP32 reports.
_INT8_MAX = 127


def _seed_for(room: str, extra: int = 0) -> int:
    """Stable seed derived from the room name, independent of PYTHONHASHSEED."""

    digest = hashlib.sha256(f"{room}:{extra}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def room_channel(
    room: str,
    n_subcarriers: int = LLTF_SUBCARRIERS,
    n_paths: int = 5,
) -> np.ndarray:
    """Frequency response of the multipath channel of ``room``.

    Returns a complex array of length ``n_subcarriers``.
    """

    if n_subcarriers < 2:
        raise ValueError("n_subcarriers must be >= 2")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = np.random.default_rng(_seed_for(room))
    delays = rng.uniform(0.0, n_subcarriers / 4.0, size=n_paths)
    gains = rng.uniform(0.3, 1.0, size=n_paths) * np.exp(2j * np.pi * rng.random(n_paths))
    gains[0] = abs(gains[0])  # line-of-sight path carries no extra phase

    k = np.arange(n_subcarriers)
    response = np.zeros(n_subcarriers, dtype=complex)
    for gain, delay in zip(gains, delays):
        response += gain * np.exp(-2j * np.pi * k * delay / n_subcarriers)
    return response


def simulate_lines(
    room: str,
    n_packets: int = 200,
    *,
    n_subcarriers: int = LLTF_SUBCARRIERS,
    rssi: int = -55,
    noise_floor: int = -92,
    noise_level: float = 0.05,
    mac: str = "3C:71:BF:6D:2A:78",
    channel: int = 1,
    packet_interval: float = 0.01,
    seed: int | None = None,
) -> list[str]:
    """Generate ``n_packets`` CSI_DATA lines for ``room``.

    ``noise_level`` is the per-packet noise amplitude relative to the mean
    channel magnitude. Higher values make rooms harder to tell apart.
    """

    if n_packets < 1:
        raise ValueError("n_packets must be >= 1")
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")

    rng = np.random.default_rng(_seed_for(room, 1) if seed is None else seed)
    response = room_channel(room, n_subcarriers)

    # Scale so the strongest subcarrier lands near full scale of the int8 buffer.
    peak = np.abs(response).max()
    response = response * (_INT8_MAX * 0.75 / (peak if peak > 0 else 1.0))
    sigma = noise_level * np.abs(response).mean()

    if n_subcarriers == LLTF_SUBCARRIERS:
        active = np.zeros(n_subcarriers, dtype=bool)
        active[list(LLTF_DATA_SUBCARRIERS)] = True
    else:
        active = np.ones(n_subcarriers, dtype=bool)

    k = np.arange(n_subcarriers)
    lines: list[str] = []
    base_timestamp = 1_000.0

    for index in range(n_packets):
        noise = rng.normal(0.0, sigma, n_subcarriers) + 1j * rng.normal(0.0, sigma, n_subcarriers)
        # Sampling time offset: a random linear phase ramp, as on real hardware.
        sto = rng.uniform(-0.5, 0.5)
        packet = (response + noise) * np.exp(-2j * np.pi * k * sto / n_subcarriers)
        packet[~active] = 0

        real = np.clip(np.rint(packet.real), -128, 127).astype(int)
        imag = np.clip(np.rint(packet.imag), -128, 127).astype(int)

        interleaved = np.empty(2 * n_subcarriers, dtype=int)
        interleaved[0::2] = imag
        interleaved[1::2] = real

        local_timestamp = int((base_timestamp + index * packet_interval) * 1_000_000)
        real_timestamp = base_timestamp + index * packet_interval
        packet_rssi = int(rssi + rng.integers(-2, 3))

        fields = [
            "CSI_DATA", "AP", mac, str(packet_rssi), "11", "1", "0", "1", "1", "1",
            "0", "0", "0", "0", str(noise_floor), "0", str(channel), "1",
            str(local_timestamp), "0", "101", "0", "0", f"{real_timestamp:.6f}",
            str(2 * n_subcarriers),
        ]
        body = " ".join(str(value) for value in interleaved)
        lines.append(",".join(fields) + f",[{body} ]")

    return lines


def write_simulated_capture(path: str | Path, room: str, n_packets: int = 200, **kwargs) -> Path:
    """Write a simulated capture for ``room`` to ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = simulate_lines(room, n_packets, **kwargs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def simulate_dataset(
    root: str | Path,
    rooms: Sequence[str] = ("kitchen", "bedroom", "hallway"),
    captures_per_room: int = 3,
    packets_per_capture: int = 200,
    **kwargs,
) -> list[Path]:
    """Populate ``root`` with ``data/<room>/capture_NN.csv`` files."""

    root = Path(root)
    written: list[Path] = []
    for room in rooms:
        for index in range(captures_per_room):
            path = root / room / f"capture_{index:02d}.csv"
            # A distinct seed per capture, so captures are not identical copies.
            written.append(
                write_simulated_capture(
                    path,
                    room,
                    packets_per_capture,
                    seed=_seed_for(room, index + 10),
                    **kwargs,
                )
            )
    return written
