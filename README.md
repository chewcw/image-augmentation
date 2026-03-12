# Image Augmentation Script

Single-image augmentation for vision datasets. The script reads an input image,
applies augmentations from a JSON config, and writes an output image. If
`--output` is omitted, the input file is overwritten.

## Usage

```bash
python augment_image.py --input ok.jpg --config SAMPLE_AUGMENT_CONFIG.json --output out.jpg
```

Deterministic output with a fixed seed:

```bash
python augment_image.py --input ok.jpg --config SAMPLE_AUGMENT_CONFIG.json --output out.jpg --seed 42
```

## Config Format (JSON)

All blocks are optional. Missing blocks are skipped.

Key ranges are `{ "min": X, "max": Y }`. If `min == max`, the value is fixed.

### rotation

```json
{
  "rotation": {
    "degrees": { "min": -5, "max": 5 },
    "fill_color": [0, 0, 0]
  }
}
```

### brightness

```json
{
  "brightness": {
    "factor": { "min": 0.9, "max": 1.1 }
  }
}
```

### blur

```json
{
  "blur": {
    "kernel": { "min": 1, "max": 5 },
    "sigma": { "min": 0.0, "max": 1.0 }
  }
}
```

`kernel` is forced to an odd integer >= 1.

### color_variation (HSV range masks)

```json
{
  "color_variation": {
    "ranges": [
      {
        "hsv_min": [0, 80, 50],
        "hsv_max": [10, 255, 255],
        "hue_shift": { "min": -5, "max": 5 },
        "sat_mult": { "min": 0.9, "max": 1.1 },
        "val_mult": { "min": 0.9, "max": 1.1 }
      }
    ]
  }
}
```

Notes:
- HSV uses OpenCV ranges: H 0–179, S/V 0–255.
- Multiple ranges can be provided; each is applied in order.

### object_removal (mask + inpaint)

```json
{
  "object_removal": {
    "mask_path": "masks/remove.png",
    "strategy": "telea",
    "radius": { "min": 3, "max": 7 },
    "threshold": 128,
    "invert_mask": false,
    "neighbor_ring_px": { "min": 3, "max": 9 },
    "neighbor_fill_mode": "median",
    "blur_kernel": { "min": 11, "max": 21 }
  }
}
```

Notes:
- `mask_path` can be absolute or relative to the config file location.
- Mask is grayscale; pixels >= `threshold` are removed (inpainted).
- `invert_mask` flips the removal region.
- `strategy` options: `telea`, `ns`, `neighbor_fill`, `blur_fill`.
- `telea` (default) and `ns` use OpenCV inpainting; `radius` applies.
- `neighbor_fill` fills with mean/median color from a ring around the mask; `neighbor_ring_px` controls ring thickness; `neighbor_fill_mode` is `mean` or `median`.
- `blur_fill` replaces the masked region with a heavily blurred version of the image; `blur_kernel` applies.

## HSV Range Picker GUI

Select an ROI on an image and output a ready-to-paste `color_variation` range.

```bash
python hsv_range_picker.py --input ok.jpg
```

Optional downsample for faster selection on large images:

```bash
python hsv_range_picker.py --input ok.jpg --sample 2
```

Brush mode (paint pixels to sample):

```bash
python hsv_range_picker.py --input ok.jpg --mode brush --brush 12
```

Brush controls:
- Middle mouse or Shift + left mouse: paint.
- Right mouse: erase.
- Ctrl + left mouse: pan.
- `+` / `-`: zoom in / out.
- `[` / `]`: decrease / increase brush size.
- `r`: reset mask.
- `s`: confirm selection.
- `q` or `ESC`: quit without selection.

Sample output:

```json
{
  "hsv_min": [0, 80, 50],
  "hsv_max": [10, 255, 255],
  "hue_shift": { "min": -5, "max": 5 },
  "sat_mult": { "min": 0.9, "max": 1.1 },
  "val_mult": { "min": 0.9, "max": 1.1 }
}
```

## Object Removal Mask Painter GUI

Paint a freehand mask for object removal (white = remove, black = keep).

```bash
python mask_painter.py --input ok.jpg --output masks/remove.png
```

Tips:
- Middle mouse or Shift + left mouse: paint.
- Right mouse: erase.
- Ctrl + left mouse: pan.
- `+` / `-`: zoom in / out.
- `[` / `]`: decrease / increase brush size.
- `r`: reset mask.
- `s`: save mask and exit.
- `q` or `ESC`: quit without saving.

## Output

The script prints a single line on success:

```
Saved augmented image to <path>
```
