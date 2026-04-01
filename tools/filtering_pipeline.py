#!/usr/bin/env python3

# This script provides `qalign_score`, `shannon_entropy`, and `flat_percentage`.
# For `aesthetics_score`, please use ArtiMuse and follow the official repository examples:
# https://github.com/thunderbolt215/ArtiMuse.git

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = 933120000

FLAT_PATCH_SIZE = 240
FLAT_VARIANCE_THRESHOLD = 800.0


def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def calculate_shannon_entropy(image: np.ndarray) -> float:
    from skimage.measure import shannon_entropy
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(shannon_entropy(gray))


def get_flat_percentage(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    flat_patches = 0
    total_patches = 0

    for y in range(0, height, FLAT_PATCH_SIZE):
        for x in range(0, width, FLAT_PATCH_SIZE):
            patch = gray[y:y + FLAT_PATCH_SIZE, x:x + FLAT_PATCH_SIZE]
            if patch.shape != (FLAT_PATCH_SIZE, FLAT_PATCH_SIZE):
                continue

            total_patches += 1
            sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            if magnitude.var() < FLAT_VARIANCE_THRESHOLD:
                flat_patches += 1

    if total_patches == 0:
        return 0.0
    return float(flat_patches / total_patches)


def get_qalign_score(image_path: str) -> float:
    import pyiqa
    import torch

    device = "cuda"
    qalign_model = pyiqa.create_metric("qalign", device=device)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_array = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        score = qalign_model(image_tensor)
    return float(score.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    image = load_image(str(image_path))
    result = {
        "qalign_score": get_qalign_score(str(image_path)),
        "shannon_entropy": calculate_shannon_entropy(image),
        "flat_percentage": get_flat_percentage(image),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("This script provides qalign_score, shannon_entropy, and flat_percentage. For aesthetics_score, please use ArtiMuse and follow the official repository examples: https://github.com/thunderbolt215/ArtiMuse.git")


if __name__ == "__main__":
    main()
