from PIL import Image
import os
from typing import Tuple

SUPPORTED_LOSSLESS_FORMATS = {".png", ".tiff", ".tif"}

def validate_image_format(filepath: str) -> None:
    """
    Ensures output file is a lossless format (PNG or TIFF). Rejects lossy JPEG formats.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_LOSSLESS_FORMATS:
        raise ValueError(
            f"Unsupported or lossy format '{ext}'. Strict preservation requires lossless formats: {SUPPORTED_LOSSLESS_FORMATS}"
        )

def load_image_safely(filepath: str) -> Tuple[Image.Image, dict]:
    """
    Loads an image while capturing immutable metadata (dimensions, mode, format).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")

    with Image.open(filepath) as img:
        img_copy = img.copy()
        meta = {
            "path": filepath,
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
        }
    return img_copy, meta

def save_lossless_image(image: Image.Image, output_path: str) -> None:
    """
    Saves image strictly in a lossless format with optimal compression.
    """
    validate_image_format(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".png":
        image.save(output_path, format="PNG", compress_level=4)
    elif ext in {".tiff", ".tif"}:
        image.save(output_path, format="TIFF", compression="tiff_deflate")
