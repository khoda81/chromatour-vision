# Chromatour Vision

Color calibration, correction, and swatch extraction for [Chromatour](https://github.com/khoda81/chromatour).

The goal is to turn ordinary photos of physical colors—pencils, markers, paints, yarn, swatches, and similar objects—into perceptually meaningful colors suitable for sorting in OKLab space.

## Current status

The project is currently focused on the color-calibration stage of the pipeline. The working prototype:

1. Loads an image and applies its EXIF orientation.
2. Converts images with embedded ICC profiles to sRGB; otherwise assumes sRGB.
3. Linearizes the rendered sRGB values.
4. Lets the user select a neutral gray or white reference patch.
5. Estimates the source white from that patch.
6. Applies Bradford chromatic adaptation to D65.
7. Converts the corrected XYZ image to OKLab for downstream measurement.

The corrected preview is for inspection only. Measurement data stays in floating-point XYZ/OKLab and is not clipped to the display gamut.

## Usage

```bash
uv sync
uv run -m color_cv path/to/image.jpg
```

Click a gray or white region when prompted. The script reports the selected point, estimated source white, and Bradford adaptation matrix, then shows the original and corrected images.

Current artifacts are written to `artifacts/`:

- `corrected_xyz.npy` — chromatically adapted XYZ image
- `corrected_oklab.npy` — OKLab measurement image
- `corrected_preview.png` — clipped sRGB preview for visual inspection

## Direction

The intended pipeline is:

```text
photo / camera
    ↓
color management + white balance
    ↓
robust color-region detection
    ↓
per-region color estimation
    ↓
confidence + visual overlays
    ↓
OKLab colors for Chromatour
```

Automatic neutral-reference estimation, segmentation, robust swatch extraction, and browser/Wasm deployment are still experimental.

## License

MIT
