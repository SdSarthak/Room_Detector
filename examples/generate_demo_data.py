"""Populate ``data/`` with simulated CSI captures so the pipeline can be tried
without any ESP32 hardware.

    python examples/generate_demo_data.py --rooms kitchen bedroom hallway
    python -m room_detector train
    python -m room_detector predict --file data/kitchen/capture_00.csv

The accuracy you get on this data says nothing about real-world accuracy: the
simulated rooms are cleanly separable by construction. It exists to exercise
the code path end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout, without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from room_detector.simulate import simulate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"),
                        help="output directory (default: data)")
    parser.add_argument("--rooms", nargs="+", default=["kitchen", "bedroom", "hallway"],
                        help="room labels to simulate")
    parser.add_argument("--captures", type=int, default=4,
                        help="captures per room (default: 4)")
    parser.add_argument("--packets", type=int, default=300,
                        help="CSI packets per capture (default: 300)")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="per-packet noise level, 0.05 = easy, 0.4 = hard")
    args = parser.parse_args()

    paths = simulate_dataset(
        args.out,
        rooms=args.rooms,
        captures_per_room=args.captures,
        packets_per_capture=args.packets,
        noise_level=args.noise,
    )
    print(f"wrote {len(paths)} simulated captures under {args.out}/")
    for path in paths:
        print(f"  {path}")
    print("\nnext: python -m room_detector train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
