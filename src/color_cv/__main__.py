from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms, ImageOps

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


def xyz_to_oklab(xyz):
    lms = xyz @ XYZ_TO_LMS_OKLAB.T
    lms_root = np.cbrt(lms)
    return lms_root @ LMS_TO_OKLAB.T


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


def white_balance_from_neutral(linear_rgb, patch):
    xyz = rgb_to_xyz(linear_rgb)

    # Estimate the color of the neutral surface robustly.
    patch_xyz = xyz[patch].reshape(-1, 3)
    neutral_xyz = np.median(patch_xyz, axis=0)

    # Brightness of the gray object is irrelevant.
    source_white = neutral_xyz / neutral_xyz[1]

    adaptation = chromatic_adaptation_matrix(
        source_white,
        D65,
    )

    corrected_xyz = xyz @ adaptation.T

    return corrected_xyz, source_white, adaptation


def main(path):
    ARTIFACTS = Path("artifacts")
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

    _fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(srgb)
    axes[0].set_title("Original")

    axes[1].imshow(preview)
    axes[1].set_title("Chromatically adapted → D65")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Keep the floating-point measurement representation.
    np.save(ARTIFACTS / "corrected_xyz.npy", corrected_xyz)
    np.save(ARTIFACTS / "corrected_oklab.npy", oklab)

    Image.fromarray(np.round(preview * 255).astype(np.uint8)).save(
        ARTIFACTS / "corrected_preview.png"
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
