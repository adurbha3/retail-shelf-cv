#!/usr/bin/env python
"""Render a synthetic shelf image with ground-truth slot boxes (green = occupied,
red = empty) as a PNG, no model required -- useful for sanity-checking the generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw
from retailshelf.synthetic import generate_shelf_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--occupancy-rate", type=float, default=0.7)
    parser.add_argument("--out", type=str, default="assets/shelf_preview.png")
    args = parser.parse_args()

    shelf = generate_shelf_image(occupancy_rate=args.occupancy_rate, seed=args.seed)
    img = shelf.image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for slot in shelf.slots:
        color = (40, 180, 80) if slot.occupied else (200, 40, 40)
        draw.rectangle(slot.bbox, outline=color, width=2)
        if slot.occupied:
            draw.rectangle(slot.price_tag_bbox, outline=(30, 90, 220), width=1)
            draw.rectangle(slot.barcode_bbox, outline=(220, 140, 20), width=1)

    img.save(args.out)
    print(f"Saved preview to {args.out}")


if __name__ == "__main__":
    main()
