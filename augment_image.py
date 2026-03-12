#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply configured augmentations to a single image."
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--output",
        help="Path to output image (default: overwrite input)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for determinism")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")
    return data


def rand_range(rng: random.Random, spec: dict[str, Any], default: float) -> float:
    if not isinstance(spec, dict):
        return default
    if "min" not in spec and "max" not in spec:
        return default
    min_v = spec.get("min", default)
    max_v = spec.get("max", default)
    try:
        min_f = float(min_v)
        max_f = float(max_v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid range values: {spec}") from exc
    if min_f == max_f:
        return min_f
    if min_f > max_f:
        min_f, max_f = max_f, min_f
    return rng.uniform(min_f, max_f)


def rand_int_range(rng: random.Random, spec: dict[str, Any], default: int) -> int:
    value = rand_range(rng, spec, float(default))
    return int(round(value))


def ensure_odd_positive(value: int) -> int:
    if value < 1:
        value = 1
    if value % 2 == 0:
        value += 1
    return value


def apply_rotation(image: np.ndarray, cfg: dict[str, Any], rng: random.Random) -> np.ndarray:
    degrees = rand_range(rng, cfg.get("degrees", {}), 0.0)
    if abs(degrees) < 1e-6:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    fill_color = cfg.get("fill_color", [0, 0, 0])
    if not (isinstance(fill_color, list) and len(fill_color) == 3):
        fill_color = [0, 0, 0]
    border_value = tuple(int(max(0, min(255, v))) for v in fill_color)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def apply_brightness(
    image: np.ndarray, cfg: dict[str, Any], rng: random.Random
) -> np.ndarray:
    factor = rand_range(rng, cfg.get("factor", {}), 1.0)
    if abs(factor - 1.0) < 1e-6:
        return image
    out = image.astype(np.float32) * factor
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def apply_blur(image: np.ndarray, cfg: dict[str, Any], rng: random.Random) -> np.ndarray:
    kernel = rand_int_range(rng, cfg.get("kernel", {}), 1)
    kernel = ensure_odd_positive(kernel)
    if kernel == 1:
        return image
    sigma = rand_range(rng, cfg.get("sigma", {}), 0.0)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=float(sigma))


def normalize_hsv_range(value: list[Any]) -> tuple[int, int, int]:
    if not (isinstance(value, list) and len(value) == 3):
        raise ValueError("HSV ranges must be lists of 3 integers.")
    h = int(max(0, min(179, int(value[0]))))
    s = int(max(0, min(255, int(value[1]))))
    v = int(max(0, min(255, int(value[2]))))
    return h, s, v


def apply_color_variation(
    image: np.ndarray, cfg: dict[str, Any], rng: random.Random
) -> np.ndarray:
    ranges = cfg.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.int16)
    s = s.astype(np.float32)
    v = v.astype(np.float32)

    for item in ranges:
        if not isinstance(item, dict):
            continue
        hsv_min = normalize_hsv_range(item.get("hsv_min", [0, 0, 0]))
        hsv_max = normalize_hsv_range(item.get("hsv_max", [179, 255, 255]))
        lower = np.array(hsv_min, dtype=np.uint8)
        upper = np.array(hsv_max, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if mask is None:
            continue

        hue_shift = rand_int_range(rng, item.get("hue_shift", {}), 0)
        sat_mult = rand_range(rng, item.get("sat_mult", {}), 1.0)
        val_mult = rand_range(rng, item.get("val_mult", {}), 1.0)

        m = mask > 0
        if hue_shift != 0:
            h[m] = (h[m] + hue_shift) % 180
        if abs(sat_mult - 1.0) > 1e-6:
            s[m] = np.clip(s[m] * sat_mult, 0, 255)
        if abs(val_mult - 1.0) > 1e-6:
            v[m] = np.clip(v[m] * val_mult, 0, 255)

    out_hsv = cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)))
    return cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)


def apply_object_removal(
    image: np.ndarray,
    cfg: dict[str, Any],
    rng: random.Random,
    config_path: Path,
) -> np.ndarray:
    mask_path = cfg.get("mask_path")
    if not isinstance(mask_path, str) or not mask_path:
        raise ValueError("object_removal.mask_path must be a non-empty string.")

    mask_file = Path(mask_path)
    if not mask_file.is_absolute():
        mask_file = (config_path.parent / mask_file).resolve()

    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read object removal mask: {mask_file}")

    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    threshold = cfg.get("threshold", 128)
    try:
        threshold_val = int(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("object_removal.threshold must be an integer.") from exc
    threshold_val = max(0, min(255, threshold_val))

    invert_mask = bool(cfg.get("invert_mask", False))
    mask_bin = (mask >= threshold_val).astype(np.uint8) * 255
    if invert_mask:
        mask_bin = 255 - mask_bin

    radius = rand_int_range(rng, cfg.get("radius", {}), 3)
    if radius < 1:
        radius = 1

    strategy = cfg.get("strategy", "telea")
    if not isinstance(strategy, str):
        strategy = "telea"
    strategy = strategy.strip().lower()

    if strategy == "telea":
        return cv2.inpaint(image, mask_bin, float(radius), cv2.INPAINT_TELEA)
    if strategy == "ns":
        return cv2.inpaint(image, mask_bin, float(radius), cv2.INPAINT_NS)
    if strategy == "neighbor_fill":
        # Fill with mean/median color from a ring around the mask.
        ring_px = rand_int_range(rng, cfg.get("neighbor_ring_px", {}), 5)
        ring_px = max(1, ring_px)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
        )
        dilated = cv2.dilate(mask_bin, kernel, iterations=1)
        ring = cv2.subtract(dilated, mask_bin)
        ring_mask = ring > 0
        if not np.any(ring_mask):
            return image

        fill_mode = cfg.get("neighbor_fill_mode", "mean")
        if not isinstance(fill_mode, str):
            fill_mode = "mean"
        fill_mode = fill_mode.strip().lower()
        ring_pixels = image[ring_mask]
        if ring_pixels.size == 0:
            return image
        if fill_mode == "median":
            fill_color = np.median(ring_pixels, axis=0)
        else:
            fill_color = np.mean(ring_pixels, axis=0)
        out = image.copy()
        out[mask_bin > 0] = np.clip(fill_color, 0, 255).astype(np.uint8)
        return out
    if strategy == "blur_fill":
        # Fill by heavy blur of the surrounding context.
        blur_kernel = rand_int_range(rng, cfg.get("blur_kernel", {}), 15)
        blur_kernel = ensure_odd_positive(blur_kernel)
        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        out = image.copy()
        out[mask_bin > 0] = blurred[mask_bin > 0]
        return out

    raise ValueError(
        "object_removal.strategy must be one of: telea, ns, neighbor_fill, blur_fill."
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        config = load_json(config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image: {input_path}", file=sys.stderr)
        return 2

    if isinstance(config.get("rotation"), dict):
        image = apply_rotation(image, config["rotation"], rng)
    if isinstance(config.get("brightness"), dict):
        image = apply_brightness(image, config["brightness"], rng)
    if isinstance(config.get("blur"), dict):
        image = apply_blur(image, config["blur"], rng)
    if isinstance(config.get("color_variation"), dict):
        image = apply_color_variation(image, config["color_variation"], rng)
    if isinstance(config.get("object_removal"), dict):
        try:
            image = apply_object_removal(
                image, config["object_removal"], rng, config_path
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    output_path = Path(args.output) if args.output else input_path
    if not cv2.imwrite(str(output_path), image):
        print(f"Failed to write output: {output_path}", file=sys.stderr)
        return 2

    print(f"Saved augmented image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
