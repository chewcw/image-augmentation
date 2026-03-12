#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 at a fixed interval."
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

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Could not determine FPS; try --every-frames.", file=sys.stderr)
        return 2

    if args.every_frames is not None:
        frame_interval = max(1, args.every_frames)
    else:
        frame_interval = max(1, int(round(fps * args.every_seconds)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{args.prefix}_{saved_idx:0{args.zero_pad}d}.{args.format}"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(
        f"Done. Saved {saved_idx} frames to {output_dir} "
        f"(interval: {frame_interval} frames, fps: {fps:.3f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
