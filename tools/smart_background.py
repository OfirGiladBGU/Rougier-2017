"""Batch convert images to grayscale and whiten backgrounds with a simple smart-background heuristic.

- Samples four edge points (midpoints of each side) to guess the background value via majority vote.
- Any pixel within the tolerance of that background value is set to white.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_background_value(img_np: np.ndarray) -> int:
    """Return the modal value from four edge midpoints."""
    height, width = img_np.shape
    coords = [
        (0, width // 2),  # top edge midpoint
        (height - 1, width // 2),  # bottom edge midpoint
        (height // 2, 0),  # left edge midpoint
        (height // 2, width - 1),  # right edge midpoint
    ]
    samples = [int(img_np[y, x]) for y, x in coords]
    counts = np.bincount(np.array(samples), minlength=256)
    return int(np.argmax(counts))


def whiten_background(img_np: np.ndarray, bg_value: int, tolerance: int) -> np.ndarray:
    """Return a copy where pixels near bg_value are set to white."""
    tol = max(0, int(tolerance))
    diff = np.abs(img_np.astype(np.int16) - int(bg_value))
    mask = diff <= tol
    output = img_np.copy()
    output[mask] = 255
    return output


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def process_image(path: Path, output_dir: Path, tolerance: int) -> None:
    image = Image.open(path).convert("L")
    img_np = np.array(image)
    bg_value = find_background_value(img_np)
    cleaned_np = whiten_background(img_np, bg_value, tolerance)
    cleaned = Image.fromarray(cleaned_np.astype(np.uint8), mode="L")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.save(output_dir / path.name)


def main() -> None:
    """
    Batch process images in a folder to whiten backgrounds using a smart-background heuristic.
    """
    # Edit these three lines as needed.
    input_dir = Path("fill50k-gs/original-rgb")
    output_dir = Path("fill50k-gs/original")
    tolerance = 0  # 0 = exact match; increase to allow variation.

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    for path in tqdm(images, desc="Processing", unit="img"):
        process_image(path, output_dir, tolerance)

    print(f"Processed {len(images)} image(s) -> {output_dir}")


if __name__ == "__main__":
    main()
