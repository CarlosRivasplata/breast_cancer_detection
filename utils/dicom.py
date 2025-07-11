import pydicom
import numpy as np
from PIL import Image


def load_dicom_image(path, out_mode="L"):
    """
    Read a DICOM file and return a PIL Image.

    Args:
        path (str): Path to the .dcm file.
        out_mode (str): Pillow mode to convert to (“L” for 8‑bit grayscale, “RGB”, etc.).

    Returns:
        PIL.Image.Image: The loaded image.
    """
    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array.astype(np.float32)

        vmin, vmax = np.nanmin(pixel_array), np.nanmax(pixel_array)
        if vmax == vmin:
            scaled = np.full_like(pixel_array, 128, dtype=np.uint8)
        else:
            scaled = ((pixel_array - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        img = Image.fromarray(scaled)
        img = img.convert(out_mode)

        return img

    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM at {path}: {e}")

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path, supporting both DICOM and standard image formats.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image.Image: Loaded image in RGB format
        
    Raises:
        RuntimeError: If image loading fails
    """
    if str(image_path).lower().endswith(".dcm"):
        ds = pydicom.dcmread(image_path)
        image_array = ds.pixel_array.astype(np.float32)

        # Normalize to [0, 255]
        image_array -= image_array.min()
        image_array /= (image_array.max() + 1e-5)
        image_array *= 255.0
        image_array = image_array.clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(image_array).convert("RGB")
        return pil_img
    else:
        return Image.open(image_path).convert("RGB")
