from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

from constants import CLEANED_IMAGES_ABS_PATH, TEXT_CLEANED_IMAGES_ABS_PATH


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def detect_best_ocr_result(image: Image.Image, reader, conf_threshold=0.5):
    variants = {
        "original": image,
        "flipped": image.transpose(Image.FLIP_LEFT_RIGHT),
        "rot45": image.rotate(45, expand=True),
        "rot135": image.rotate(135, expand=True),
    }

    best_result = []
    best_score = 0
    best_transform = "original"

    for name, variant in variants.items():
        ocr_result = reader.readtext(np.array(variant))
        score = sum(conf for _, _, conf in ocr_result if conf >= conf_threshold)
        if score > best_score:
            best_result = ocr_result
            best_score = score
            best_transform = name

    return best_result, best_transform


def crop_fixed_margins(image: Image.Image, crop_px: int = 60) -> Image.Image:
    w, h = image.size
    left = crop_px
    right = w - crop_px
    top = crop_px
    bottom = h - crop_px

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


def mask_text(image: Image.Image, reader, fill=(0, 0, 0), draw_outline=False):
    results, transform = detect_best_ocr_result(image, reader)

    # Apply same transformation to align coordinates
    if transform == "flipped":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == "rot45":
        image = image.rotate(45, expand=True)
    elif transform == "rot135":
        image = image.rotate(135, expand=True)

    draw = ImageDraw.Draw(image)
    text_found = False

    for (bbox, text, conf) in results:
        if conf < 0.5:
            continue
        text_found = True
        pts = [tuple(map(int, point)) for point in bbox]
        x_coords, y_coords = zip(*pts)
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        pad_x = int(bbox_width * 0.25)
        pad_y = int(bbox_height * 0.25)

        x_min = max(min(x_coords) - pad_x, 0)
        x_max = min(max(x_coords) + pad_x, image.width)
        y_min = max(min(y_coords) - pad_y, 0)
        y_max = min(max(y_coords) + pad_y, image.height)

        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=fill)
        if draw_outline:
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=1)

    # Undo transformation
    if transform == "flipped":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == "rot45":
        image = image.rotate(-45, expand=True)
    elif transform == "rot135":
        image = image.rotate(-135, expand=True)

    return image, text_found


def clean_images_in_directory(source_root: str, output_root: str, use_gpu: bool = False, crop_px: int = 60):
    supported_exts = {".png", ".jpg", ".jpeg"}
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_folder = source_root.name
    log_filename = f"log_{timestamp}_{source_folder}.csv"
    log_path = output_root / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["file_path", "status", "error"]).to_csv(log_path, index=False)

    files_to_process = [fp for fp in source_root.rglob("*") if fp.suffix.lower() in supported_exts]
    print(f"Processing {len(files_to_process)} images...")

    reader = easyocr.Reader(['en'], gpu=use_gpu)
    results = []

    for fp in tqdm(files_to_process, desc="Processing", unit="img"):
        try:
            image = load_image(str(fp))
            masked_img, found_text = mask_text(image.copy(), reader)

            rel_path = fp.relative_to(source_root)
            output_path = (output_root / rel_path).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if found_text:
                masked_img = crop_fixed_margins(masked_img, crop_px=crop_px)
                masked_img.save(output_path, format="PNG")
                status = "processed"
            else:
                image.save(output_path, format="PNG")
                status = "copied"

            results.append({"file_path": str(fp), "status": status, "error": ""})
        except Exception as e:
            results.append({"file_path": str(fp), "status": "failed", "error": str(e)})

    pd.DataFrame(results).to_csv(log_path, mode='a', header=False, index=False)
    print(f"\nLog saved to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and mask text in images (PNG/JPG), saving output.")
    parser.add_argument("--source", "-s", required=False, type=str,
                        help="Path to the directory containing preconverted images.")
    parser.add_argument("--output", "-o", required=False, type=str,
                        help="Path to the output directory for cleaned images.")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU usage for OCR.")
    parser.add_argument("--crop", type=int, default=60, help="Pixels to crop from borders after masking text.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_root = args.source if args.source else CLEANED_IMAGES_ABS_PATH
    output_root = args.output if args.output else TEXT_CLEANED_IMAGES_ABS_PATH

    clean_images_in_directory(
        source_root=source_root,
        output_root=output_root,
        use_gpu=args.gpu,
        crop_px=args.crop
    )
