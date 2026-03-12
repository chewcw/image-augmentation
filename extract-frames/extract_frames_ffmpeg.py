#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 using ffmpeg."
    )
    parser.add_argument("input", help="Path to input MP4")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="frames",
        help="Directory for extracted images (default: frames)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="Extract every N seconds (default: 1.0)",
    )
    group.add_argument(
        "--every-frames",
        type=int,
        help="Extract every N frames (overrides --every-seconds)",
    )
    parser.add_argument(
        "--format",
        default="jpg",
        choices=["jpg", "png", "bmp", "tif", "tiff", "webp"],
        help="Image format/extension (default: jpg)",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Filename prefix (default: frame)",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=6,
        help="Zero padding for frame number (default: 6)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = output_dir / f"{args.prefix}_%0{args.zero_pad}d.{args.format}"

    if args.every_frames is not None:
        vf = f"select='not(mod(n\\,{max(1, args.every_frames)}))'"
        vsync = "vfr"
    else:
        interval = max(0.0001, args.every_seconds)
        vf = f"fps=1/{interval}"
        vsync = "vfr"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-vsync",
        vsync,
        str(out_pattern),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print(f"Done. Saved frames to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
