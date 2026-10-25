"""Seeded synthetic retail shelf image generator.

Each image is a shelf of grid slots; some slots hold a product (a colored
facing with a price tag and barcode), others are empty (out-of-stock gaps) --
mirroring the shelf-availability problem a real shelf-scanning robot has to
solve from noisy, cluttered imagery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from retailshelf.barcode import render_barcode, barcode_width_px, BAR_HEIGHT_PX

IMG_WIDTH = 384
IMG_HEIGHT = 256
ROWS = 2
COLS = 4
SHELF_BG = (222, 214, 198)

PRODUCT_COLORS = [
    (200, 60, 60), (60, 120, 200), (60, 170, 90), (210, 160, 40),
    (150, 90, 180), (220, 110, 150), (90, 90, 90), (40, 160, 160),
]


@dataclass
class Slot:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    occupied: bool
    price: float | None = None
    sku: int | None = None
    price_tag_bbox: tuple[int, int, int, int] | None = None
    barcode_bbox: tuple[int, int, int, int] | None = None


@dataclass
class ShelfImage:
    image: Image.Image
    slots: list[Slot]


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _grid_cells(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    cell_w = IMG_WIDTH // COLS
    cell_h = IMG_HEIGHT // ROWS
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            jitter_x = int(rng.integers(-4, 5))
            jitter_y = int(rng.integers(-4, 5))
            x0 = c * cell_w + 6 + jitter_x
            y0 = r * cell_h + 6 + jitter_y
            x1 = (c + 1) * cell_w - 6 + jitter_x
            y1 = (r + 1) * cell_h - 6 + jitter_y
            cells.append((x0, y0, x1, y1))
    return cells


def generate_shelf_image(
    occupancy_rate: float = 0.7,
    noise_sigma: float = 6.0,
    blur_radius: float = 0.4,
    seed: int | None = None,
) -> ShelfImage:
    """Generate one synthetic shelf image with ground-truth slot annotations."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=SHELF_BG)
    draw = ImageDraw.Draw(img)
    font = _default_font(13)

    slots: list[Slot] = []
    for bbox in _grid_cells(rng):
        x0, y0, x1, y1 = bbox
        occupied = bool(rng.random() < occupancy_rate)

        if not occupied:
            draw.rectangle(bbox, outline=(180, 172, 158), width=1)
            slots.append(Slot(bbox=bbox, occupied=False))
            continue

        color = PRODUCT_COLORS[int(rng.integers(0, len(PRODUCT_COLORS)))]
        draw.rectangle(bbox, fill=color, outline=(40, 40, 40), width=1)

        price = round(float(rng.uniform(0.99, 29.99)), 2)
        sku = int(rng.integers(0, 10**6))

        tag_w, tag_h = 40, 16
        tag_x0, tag_y0 = x0 + 3, y1 - tag_h - 3
        tag_bbox = (tag_x0, tag_y0, tag_x0 + tag_w, tag_y0 + tag_h)
        draw.rectangle(tag_bbox, fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((tag_x0 + 3, tag_y0 + 2), f"${price:05.2f}", fill=(0, 0, 0), font=font)

        bar_w, bar_h = barcode_width_px(), BAR_HEIGHT_PX
        scale = min((x1 - x0 - 10) / bar_w, 0.5)
        bar_img = render_barcode(sku).resize(
            (max(1, int(bar_w * scale)), max(1, int(bar_h * scale)))
        )
        bar_x0 = x0 + 3
        bar_y0 = y0 + 3
        img.paste(bar_img.convert("RGB"), (bar_x0, bar_y0))
        bar_bbox = (bar_x0, bar_y0, bar_x0 + bar_img.width, bar_y0 + bar_img.height)

        slots.append(
            Slot(
                bbox=bbox, occupied=True, price=price, sku=sku,
                price_tag_bbox=tag_bbox, barcode_bbox=bar_bbox,
            )
        )

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    if noise_sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += rng.normal(0, noise_sigma, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return ShelfImage(image=img, slots=slots)


def generate_dataset(n_images: int, seed: int | None = None, **kwargs) -> list[ShelfImage]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_images):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        out.append(generate_shelf_image(seed=sub_seed, **kwargs))
    return out
