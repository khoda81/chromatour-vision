from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
from PIL import Image, ImageCms, ImageOps
from plotly.subplots import make_subplots

# ---------- sRGB / XYZ ----------

RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)

D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055,
    )


def rgb_to_xyz(rgb):
    return rgb @ RGB_TO_XYZ.T


def xyz_to_rgb(xyz):
    return xyz @ XYZ_TO_RGB.T


# ---------- Bradford chromatic adaptation ----------

BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)

BRADFORD_INV = np.linalg.inv(BRADFORD)


def chromatic_adaptation_matrix(source_white, target_white=D65):
    source_lms = BRADFORD @ source_white
    target_lms = BRADFORD @ target_white

    scale = np.diag(target_lms / source_lms)

    return BRADFORD_INV @ scale @ BRADFORD


# ---------- OKLab ----------

XYZ_TO_LMS_OKLAB = np.array(
    [
        [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
        [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
        [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
    ]
)

LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

LMS_TO_XYZ_OKLAB = np.linalg.inv(XYZ_TO_LMS_OKLAB)
OKLAB_TO_LMS = np.linalg.inv(LMS_TO_OKLAB)


def xyz_to_oklab(xyz):
    lms = xyz @ XYZ_TO_LMS_OKLAB.T
    lms_root = np.cbrt(lms)
    return lms_root @ LMS_TO_OKLAB.T


def oklab_to_xyz(oklab):
    lms_root = np.asarray(oklab) @ OKLAB_TO_LMS.T
    lms = lms_root**3
    return lms @ LMS_TO_XYZ_OKLAB.T


def oklab_to_srgb(oklab):
    xyz = oklab_to_xyz(oklab)
    linear_rgb = xyz_to_rgb(xyz)
    return linear_to_srgb(linear_rgb)


# ---------- image loading ----------


def load_srgb(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)

    # JPEGs/PNGs can contain an ICC profile, including Display P3.
    # Normalize it to sRGB before interpreting the channel values.
    icc = image.info.get("icc_profile")

    if icc:
        source_profile = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc))
        srgb_profile = ImageCms.createProfile("sRGB")

        image = ImageCms.profileToProfile(
            image,
            source_profile,
            srgb_profile,
            outputMode="RGB",
        )
    else:
        # No profile: assume the file means sRGB.
        image = image.convert("RGB")

    return np.asarray(image, dtype=np.float64) / 255.0


def choose_neutral(image, radius=15):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title("Click a gray or white surface")
    plt.axis("off")

    [(x, y)] = plt.ginput(1)
    plt.close()

    x = round(x)
    y = round(y)

    y0 = max(0, y - radius)
    y1 = min(image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(image.shape[1], x + radius + 1)

    return (x, y), (slice(y0, y1), slice(x0, x1))


def polygon_to_mask(shape, vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least three vertices")

    height, width = shape[:2]
    x0 = max(0, int(np.floor(vertices[:, 0].min())))
    x1 = min(width, int(np.ceil(vertices[:, 0].max())) + 1)
    y0 = max(0, int(np.floor(vertices[:, 1].min())))
    y1 = min(height, int(np.ceil(vertices[:, 1].max())) + 1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Polygon is outside the image")

    yy, xx = np.mgrid[y0:y1, x0:x1]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    local_mask = MplPath(vertices).contains_points(points, radius=1e-9)
    local_mask = local_mask.reshape(y1 - y0, x1 - x0)

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = local_mask
    return mask


def choose_roi(image):
    selection = {}
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(
        "Click around one colored object; click the first point again to close the polygon"
    )
    ax.axis("off")

    def on_select(vertices):
        if len(vertices) < 3:
            return

        mask = polygon_to_mask(image.shape, vertices)
        if mask.any():
            selection["mask"] = mask
            selection["vertices"] = np.asarray(vertices)
            plt.close(fig)

    selector = PolygonSelector(ax, on_select, useblit=True)
    plt.show()
    selector.disconnect_events()

    if "mask" not in selection:
        raise RuntimeError("No ROI selected")

    return selection["mask"], selection["vertices"]


def white_balance_from_neutral(linear_rgb, patch):
    xyz = rgb_to_xyz(linear_rgb)

    # Estimate the color of the neutral surface robustly.
    patch_xyz = xyz[patch].reshape(-1, 3)
    neutral_xyz = np.median(patch_xyz, axis=0)

    # Brightness of the gray object is irrelevant.
    source_white = neutral_xyz / neutral_xyz[1]

    adaptation = chromatic_adaptation_matrix(source_white, D65)

    corrected_xyz = xyz @ adaptation.T

    return corrected_xyz, source_white, adaptation


# ---------- ROI color estimation ----------


def estimate_roi_color(oklab, mask, trim=0.1):
    pixels = oklab[mask]
    pixels = pixels[np.isfinite(pixels).all(axis=1)]

    if not len(pixels):
        raise ValueError("ROI contains no valid pixels")

    low, high = np.quantile(pixels[:, 0], [trim, 1 - trim])
    kept = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]

    return np.median(kept, axis=0), pixels, kept


def srgb_to_hex(srgb):
    rgb8 = np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    return "#{:02X}{:02X}{:02X}".format(*rgb8)


def make_overlay(preview, mask, estimated_oklab, alpha=0.5):
    overlay = preview.copy()
    estimated_srgb = np.clip(oklab_to_srgb(estimated_oklab), 0, 1)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * estimated_srgb
    return overlay


def _sample_indices(size, limit=5000):
    if size <= limit:
        return np.arange(size)
    return np.linspace(0, size - 1, limit, dtype=int)


def _rgb_strings(rgb):
    rgb8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb8]


def write_roi_diagnostics(path, pixels, roi_srgb, estimated_oklab):
    indices = _sample_indices(len(pixels))
    sample = pixels[indices]
    sample_rgb = roi_srgb.reshape(-1, 3)[indices]
    colors = _rgb_strings(sample_rgb)

    lightness = sample[:, 0]
    chroma = np.linalg.norm(sample[:, 1:], axis=1)
    safe_lightness = np.where(np.abs(lightness) > 1e-8, lightness, np.nan)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "OKLab chroma plane",
            "Shading-normalized chroma",
            "Lightness vs chroma",
        ),
    )

    marker = {"color": colors, "size": 4, "opacity": 0.45}
    hover = "L=%{customdata[0]:.4f}<br>a=%{customdata[1]:.4f}<br>b=%{customdata[2]:.4f}<extra></extra>"

    fig.add_trace(
        go.Scattergl(
            x=sample[:, 1],
            y=sample[:, 2],
            mode="markers",
            marker=marker,
            customdata=sample,
            hovertemplate=hover,
            name="ROI pixels",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=sample[:, 1] / safe_lightness,
            y=sample[:, 2] / safe_lightness,
            mode="markers",
            marker=marker,
            customdata=sample,
            hovertemplate=hover,
            name="ROI pixels",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scattergl(
            x=lightness,
            y=chroma,
            mode="markers",
            marker=marker,
            customdata=sample,
            hovertemplate=hover,
            name="ROI pixels",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    estimate_chroma = np.linalg.norm(estimated_oklab[1:])
    estimate_l = estimated_oklab[0]
    estimate_marker = {
        "color": srgb_to_hex(oklab_to_srgb(estimated_oklab)),
        "size": 14,
        "symbol": "x",
        "line": {"width": 2},
    }

    fig.add_trace(
        go.Scatter(
            x=[estimated_oklab[1]],
            y=[estimated_oklab[2]],
            mode="markers",
            marker=estimate_marker,
            name="estimate",
        ),
        row=1,
        col=1,
    )
    if abs(estimate_l) > 1e-8:
        fig.add_trace(
            go.Scatter(
                x=[estimated_oklab[1] / estimate_l],
                y=[estimated_oklab[2] / estimate_l],
                mode="markers",
                marker=estimate_marker,
                name="estimate",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[estimate_l],
            y=[estimate_chroma],
            mode="markers",
            marker=estimate_marker,
            name="estimate",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="a", row=1, col=1)
    fig.update_yaxes(title_text="b", row=1, col=1)
    fig.update_xaxes(title_text="a / L", row=1, col=2)
    fig.update_yaxes(title_text="b / L", row=1, col=2)
    fig.update_xaxes(title_text="L", row=1, col=3)
    fig.update_yaxes(title_text="C", row=1, col=3)
    fig.update_layout(title="ROI color diagnostics", height=560)
    fig.write_html(path, include_plotlyjs=True)


def main(path):
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    srgb = load_srgb(path)
    linear = srgb_to_linear(srgb)

    point, patch = choose_neutral(srgb)

    corrected_xyz, estimated_white, matrix = white_balance_from_neutral(linear, patch)

    # THIS is the representation we'll eventually measure.
    oklab = xyz_to_oklab(corrected_xyz)

    # Preview only.
    corrected_linear_rgb = xyz_to_rgb(corrected_xyz)
    corrected_srgb = linear_to_srgb(corrected_linear_rgb)

    # Do not clip measurement data.
    # Only clip the display image.
    preview = np.clip(corrected_srgb, 0, 1)

    print(f"Neutral sample: {point}")
    print(f"Estimated source white XYZ: {estimated_white}")
    print("Bradford matrix:")
    print(matrix)

    roi_mask, roi_vertices = choose_roi(preview)
    estimated_color, roi_pixels, kept_pixels = estimate_roi_color(oklab, roi_mask)
    estimated_srgb = oklab_to_srgb(estimated_color)
    overlay = make_overlay(preview, roi_mask, estimated_color)

    print(f"ROI vertices: {len(roi_vertices)}")
    print(f"ROI pixels: {len(roi_pixels):,}")
    print(f"Pixels after lightness trim: {len(kept_pixels):,}")
    print(f"Estimated OKLab: {estimated_color}")
    print(f"Estimated sRGB: {estimated_srgb}")
    print(f"Estimated hex: {srgb_to_hex(estimated_srgb)}")

    roi_srgb = preview[roi_mask]
    write_roi_diagnostics(
        artifacts / "roi_diagnostics.html",
        roi_pixels,
        roi_srgb,
        estimated_color,
    )

    _fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(preview)
    axes[0].set_title("Chromatically adapted → D65")
    axes[1].imshow(overlay)
    axes[1].set_title(f"50% estimated-color overlay — {srgb_to_hex(estimated_srgb)}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Keep the floating-point measurement representation.
    np.save(artifacts / "corrected_xyz.npy", corrected_xyz)
    np.save(artifacts / "corrected_oklab.npy", oklab)

    Image.fromarray(np.round(preview * 255).astype(np.uint8)).save(
        artifacts / "corrected_preview.png"
    )
    Image.fromarray(np.round(overlay * 255).astype(np.uint8)).save(
        artifacts / "roi_overlay.png"
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
