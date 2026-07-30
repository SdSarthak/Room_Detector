"""Command line interface: ``python -m room_detector <command>``.

Commands
--------
``collect``   record labelled CSI captures from an ESP32 over serial or stdin
``inspect``   summarise a capture file without training anything
``train``     build a dataset from ``data/`` and fit a room classifier
``evaluate``  cross-validate without writing a model
``predict``   classify a live serial stream, stdin, or a recorded capture
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from .config import Config
from .csi import CSI_MARKER, read_csi_file, records_to_arrays
from .dataset import build_dataset
from .model import RoomClassifier
from .stream import file_lines, run_stream, serial_lines, stdin_lines

__all__ = ["main", "build_parser"]


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="room_detector",
        description="ESP32 CSI based indoor room detection.",
    )
    parser.add_argument("--config", type=Path, help="YAML config file to load")
    parser.add_argument("--data-dir", type=Path, help="directory of labelled captures")
    parser.add_argument("--model-path", type=Path, help="where the trained model lives")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="record a labelled CSI capture")
    collect.add_argument("--room", required=True, help="room label, e.g. kitchen")
    collect.add_argument("--port", help="serial port (default: from config/.env)")
    collect.add_argument("--baudrate", type=int, help="serial baud rate")
    collect.add_argument("--stdin", action="store_true",
                         help="read serial output from stdin instead of opening a port")
    collect.add_argument("--seconds", type=float, help="stop after this many seconds")
    collect.add_argument("--packets", type=int, help="stop after this many CSI packets")
    collect.add_argument("--out", type=Path, help="explicit output file path")

    inspect = sub.add_parser("inspect", help="summarise a capture file")
    inspect.add_argument("path", type=Path, help="capture .csv to inspect")

    train = sub.add_parser("train", help="train a room classifier")
    train.add_argument("--algorithm",
                       choices=["random_forest", "extra_trees", "svm", "knn", "logistic"])
    train.add_argument("--window-size", type=int)
    train.add_argument("--window-step", type=int)
    train.add_argument("--no-evaluate", action="store_true",
                       help="skip cross-validation and just fit")

    evaluate = sub.add_parser("evaluate", help="cross-validate without saving a model")
    evaluate.add_argument("--algorithm",
                          choices=["random_forest", "extra_trees", "svm", "knn", "logistic"])
    evaluate.add_argument("--window-size", type=int)
    evaluate.add_argument("--window-step", type=int)

    predict = sub.add_parser("predict", help="classify a live or recorded stream")
    predict.add_argument("--port", help="serial port to read from")
    predict.add_argument("--baudrate", type=int)
    predict.add_argument("--stdin", action="store_true", help="read serial output from stdin")
    predict.add_argument("--file", type=Path, help="replay a recorded capture instead")
    predict.add_argument("--min-confidence", type=float,
                         help="suppress predictions below this confidence")

    return parser


# ----------------------------------------------------------------------
# Config assembly
# ----------------------------------------------------------------------
def _load_config(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config.from_env()

    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.model_path:
        config.model_path = Path(args.model_path)
    for attr, target in (("algorithm", config.model),):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(target, attr, value)
    for attr in ("window_size", "window_step"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config.features, attr, value)
    for attr in ("port", "baudrate", "min_confidence"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config.stream, attr, value)

    # Re-run the dataclass validation after the CLI overrides.
    config.features.__post_init__()
    config.model.__post_init__()
    config.stream.__post_init__()
    return config


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_collect(args: argparse.Namespace, config: Config) -> int:
    room = args.room.strip().replace(" ", "_")
    if not room:
        print("error: --room must not be empty", file=sys.stderr)
        return 2

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.data_dir / room / f"capture_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stdin:
        source: Iterator[str] = stdin_lines()
        origin = "stdin"
    else:
        source = serial_lines(config.stream.port, config.stream.baudrate, config.stream.timeout)
        origin = f"{config.stream.port} @ {config.stream.baudrate}"

    print(f"recording room {room!r} from {origin} -> {out_path}")
    if args.seconds:
        print(f"  stopping after {args.seconds:g}s")
    if args.packets:
        print(f"  stopping after {args.packets} packets")
    print("  press Ctrl+C to stop")

    started = time.monotonic()
    written = 0
    try:
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            for line in source:
                if CSI_MARKER not in line:
                    continue
                handle.write(line.rstrip("\r\n") + "\n")
                written += 1
                if written % 100 == 0:
                    handle.flush()
                    elapsed = time.monotonic() - started
                    print(f"\r  {written} packets ({written / max(elapsed, 1e-9):.1f}/s)",
                          end="", flush=True)
                if args.packets and written >= args.packets:
                    break
                if args.seconds and (time.monotonic() - started) >= args.seconds:
                    break
    except KeyboardInterrupt:
        print("\n  interrupted")
    except (ConnectionError, ImportError, OSError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print(f"\nwrote {written} CSI packets in {elapsed:.1f}s to {out_path}")
    if written == 0:
        print(
            "warning: no CSI packets captured. Confirm the firmware was built with "
            "CONFIG_SHOULD_COLLECT_CSI and that a transmitter is active.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_inspect(args: argparse.Namespace, config: Config) -> int:
    records = read_csi_file(args.path)
    if not records:
        print(f"no CSI packets found in {args.path}", file=sys.stderr)
        return 1

    arrays = records_to_arrays(records)
    amplitude = arrays["amplitude"]
    duration = float(arrays["timestamp"][-1] - arrays["timestamp"][0])
    dropped = int(arrays["n_dropped"])

    print(f"file:             {args.path}")
    print(f"packets:          {len(records)}"
          + (f" ({dropped} dropped: inconsistent subcarrier count)" if dropped else ""))
    print(f"subcarriers:      {amplitude.shape[1]}")
    print(f"duration:         {duration:.2f}s"
          + (f"  ({len(records) / duration:.1f} packets/s)" if duration > 0 else ""))
    print(f"rssi:             mean {arrays['rssi'].mean():.1f} dBm, "
          f"min {arrays['rssi'].min():.0f}, max {arrays['rssi'].max():.0f}")
    print(f"noise floor:      mean {arrays['noise_floor'].mean():.1f} dBm")
    print(f"amplitude:        mean {amplitude.mean():.2f}, max {amplitude.max():.2f}")
    print(f"macs:             {', '.join(sorted({r.mac for r in records}))}")
    print(f"channels:         {', '.join(str(c) for c in sorted({r.channel for r in records}))}")

    windows = max(0, (len(records) - config.features.window_size) // config.features.window_step + 1)
    print(f"training windows: {windows} "
          f"(window_size={config.features.window_size}, step={config.features.window_step})")
    return 0


def cmd_train(args: argparse.Namespace, config: Config) -> int:
    dataset = build_dataset(config)
    print(dataset.summary())
    print()

    model = RoomClassifier(config)
    if not args.no_evaluate:
        try:
            result = model.evaluate(dataset.X, dataset.y, dataset.groups)
            print(result)
        except ValueError as exc:
            print(f"skipping cross-validation: {exc}")
        print()

    model.fit_dataset(dataset)
    path = model.save(config.model_path)
    print(f"trained {config.model.algorithm} on {dataset.X.shape[0]} windows "
          f"across {len(dataset.labels)} rooms")
    print(f"saved model -> {path}")

    importances = model.feature_importances(top=10)
    if importances:
        print("\ntop features:")
        for name, value in importances:
            print(f"  {name:<24} {value:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace, config: Config) -> int:
    dataset = build_dataset(config)
    print(dataset.summary())
    print()
    model = RoomClassifier(config)
    print(model.evaluate(dataset.X, dataset.y, dataset.groups))
    return 0


def cmd_predict(args: argparse.Namespace, config: Config) -> int:
    model = RoomClassifier.load(config.model_path)
    print(f"loaded model from {config.model_path} (rooms: {', '.join(model.classes_)})")

    if args.file:
        lines: Iterator[str] = file_lines(args.file)
        print(f"replaying {args.file}")
    elif args.stdin:
        lines = stdin_lines()
        print("reading CSI from stdin")
    else:
        lines = serial_lines(config.stream.port, config.stream.baudrate, config.stream.timeout)
        print(f"reading CSI from {config.stream.port} @ {config.stream.baudrate}")

    try:
        emitted = run_stream(model, lines, config)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 0
    except (ConnectionError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if emitted == 0:
        print(
            f"no prediction produced: the stream held fewer than "
            f"{config.features.window_size} usable CSI packets",
            file=sys.stderr,
        )
        return 1
    return 0


_COMMANDS = {
    "collect": cmd_collect,
    "inspect": cmd_inspect,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "predict": cmd_predict,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(args)
        return _COMMANDS[args.command](args, config)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
