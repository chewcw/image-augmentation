#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a cutout mask using either rectangle or brush mode."
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output mask PNG")
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Optional downsample factor for faster display (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["rect", "brush"],
        default="rect",
        help="Selection mode: rect or brush (default: rect)",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=12,
        help="Initial brush radius in pixels (default: 12)",
    )
    return parser.parse_args()


class PaintState:
    def __init__(self, image: np.ndarray, brush: int, initial_mask: np.ndarray) -> None:
        self.image = image
        self.brush = max(1, int(brush))
        self.mask = initial_mask.copy()
        self.initial_mask = initial_mask.copy()
        self.drawing = False
        self.erasing = False
        self.last_pos: tuple[int, int] | None = None
        self.pan_active = False
        self.pan_last: tuple[int, int] | None = None
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def set_brush(self, value: int) -> None:
        self.brush = max(1, min(200, int(value)))

    def paint_at(self, x: int, y: int, value: int) -> None:
        cv2.circle(self.mask, (x, y), self.brush, value, thickness=-1)

    def draw_line(self, x: int, y: int, value: int) -> None:
        if self.last_pos is None:
            self.paint_at(x, y, value)
            self.last_pos = (x, y)
            return
        cv2.line(self.mask, self.last_pos, (x, y), value, thickness=self.brush * 2)
        self.last_pos = (x, y)

    def clamp_offset(self) -> None:
        h, w = self.image.shape[:2]
        view_w = w / self.zoom
        view_h = h / self.zoom
        max_x = max(0.0, w - view_w)
        max_y = max(0.0, h - view_h)
        self.offset_x = max(0.0, min(self.offset_x, max_x))
        self.offset_y = max(0.0, min(self.offset_y, max_y))

    def set_zoom(self, value: float) -> None:
        h, w = self.image.shape[:2]
        value = max(1.0, min(value, 8.0))
        if abs(value - self.zoom) < 1e-6:
            return
        center_x = self.offset_x + (w / self.zoom) * 0.5
        center_y = self.offset_y + (h / self.zoom) * 0.5
        self.zoom = value
        self.offset_x = center_x - (w / self.zoom) * 0.5
        self.offset_y = center_y - (h / self.zoom) * 0.5
        self.clamp_offset()

    def reset_mask(self) -> None:
        self.mask[:, :] = self.initial_mask


def apply_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = overlay[:, :, 2].astype(np.float32)
    red = np.clip(red + (mask.astype(np.float32) * 0.35), 0, 255)
    overlay[:, :, 2] = red.astype(np.uint8)
    return overlay


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image: {input_path}", file=sys.stderr)
        return 2

    display = image
    scale = 1.0
    if args.sample and args.sample > 1:
        scale = 1.0 / float(args.sample)
        new_w = max(1, int(round(image.shape[1] * scale)))
        new_h = max(1, int(round(image.shape[0] * scale)))
        display = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if args.mode == "rect":
        roi = cv2.selectROI(
            "Select Cutout ROI (press ENTER to confirm, ESC to cancel)",
            display,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w <= 0 or h <= 0:
            print("No ROI selected.", file=sys.stderr)
            return 2

        if scale != 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))

        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x2 - 1, y2 - 1), 255, thickness=-1)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), mask):
            print(f"Failed to write mask: {output_path}", file=sys.stderr)
            return 2
        print(f"Saved mask to {output_path}")
        return 0

    initial_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    state = PaintState(image, args.brush, initial_mask)

    display_w = display.shape[1]
    display_h = display.shape[0]

    def map_point(px: int, py: int) -> tuple[int, int]:
        view_w = image.shape[1] / state.zoom
        view_h = image.shape[0] / state.zoom
        x = state.offset_x + (px / max(1, display_w)) * view_w
        y = state.offset_y + (py / max(1, display_h)) * view_h
        x = int(round(max(0.0, min(image.shape[1] - 1, x))))
        y = int(round(max(0.0, min(image.shape[0] - 1, y))))
        return x, y

    def pan_by(dx: int, dy: int) -> None:
        view_w = image.shape[1] / state.zoom
        view_h = image.shape[0] / state.zoom
        state.offset_x -= (dx / max(1, display_w)) * view_w
        state.offset_y -= (dy / max(1, display_h)) * view_h
        state.clamp_offset()

    def on_mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        ctrl = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
        shift = (flags & cv2.EVENT_FLAG_SHIFTKEY) != 0
        if event == cv2.EVENT_LBUTTONDOWN:
            if ctrl:
                state.pan_active = True
                state.pan_last = (x, y)
                state.drawing = False
                state.erasing = False
                state.last_pos = None
                return
            if shift:
                state.drawing = True
                state.erasing = False
                state.last_pos = None
                ix, iy = map_point(x, y)
                state.paint_at(ix, iy, 255)
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            state.drawing = True
            state.erasing = False
            state.last_pos = None
            ix, iy = map_point(x, y)
            state.paint_at(ix, iy, 255)
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            state.erasing = True
            state.drawing = False
            state.last_pos = None
            ix, iy = map_point(x, y)
            state.paint_at(ix, iy, 0)
            return
        if event == cv2.EVENT_MOUSEMOVE:
            if state.pan_active and state.pan_last is not None:
                dx = x - state.pan_last[0]
                dy = y - state.pan_last[1]
                pan_by(dx, dy)
                state.pan_last = (x, y)
            elif state.drawing:
                ix, iy = map_point(x, y)
                state.draw_line(ix, iy, 255)
            elif state.erasing:
                ix, iy = map_point(x, y)
                state.draw_line(ix, iy, 0)
            return
        if event in (
            cv2.EVENT_LBUTTONUP,
            cv2.EVENT_MBUTTONUP,
            cv2.EVENT_RBUTTONUP,
        ):
            state.drawing = False
            state.erasing = False
            state.last_pos = None
            state.pan_active = False
            state.pan_last = None

    window = (
        "Cutout Mask Painter (Brush, MMB/Shift+LMB paint, RMB erase, "
        "Ctrl+LMB pan, +/- zoom)"
    )
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        overlay = apply_overlay(image, state.mask)
        view_w = int(round(image.shape[1] / state.zoom))
        view_h = int(round(image.shape[0] / state.zoom))
        x0 = int(round(state.offset_x))
        y0 = int(round(state.offset_y))
        x1 = min(image.shape[1], x0 + view_w)
        y1 = min(image.shape[0], y0 + view_h)
        view = overlay[y0:y1, x0:x1]
        overlay_disp = cv2.resize(
            view, (display_w, display_h), interpolation=cv2.INTER_AREA
        )

        cv2.putText(
            overlay_disp,
            f"Brush: {state.brush}px  Zoom: {state.zoom:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, overlay_disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            print("Cancelled without saving.", file=sys.stderr)
            return 2
        if key == ord("s"):
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), state.mask):
                cv2.destroyAllWindows()
                print(f"Failed to write mask: {output_path}", file=sys.stderr)
                return 2
            cv2.destroyAllWindows()
            print(f"Saved mask to {output_path}")
            return 0
        if key == ord("r"):
            state.reset_mask()
        if key == ord("["):
            state.set_brush(state.brush - 2)
        if key == ord("]"):
            state.set_brush(state.brush + 2)
        if key in (ord("+"), ord("=")):
            state.set_zoom(state.zoom * 1.25)
        if key in (ord("-"), ord("_")):
            state.set_zoom(state.zoom / 1.25)


if __name__ == "__main__":
    raise SystemExit(main())
