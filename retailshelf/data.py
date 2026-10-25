"""Convert synthetic ShelfImage objects into model-ready tensors and training targets."""
from __future__ import annotations

import numpy as np
import torch

from retailshelf.synthetic import ShelfImage, ROWS, COLS, IMG_WIDTH, IMG_HEIGHT
from retailshelf.detector import nominal_cell_boxes


def image_to_tensor(shelf: ShelfImage) -> torch.Tensor:
    arr = np.asarray(shelf.image, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def shelf_to_target(shelf: ShelfImage) -> torch.Tensor:
    """ShelfImage -> (ROWS, COLS, 5) target tensor: [occupied, dx0, dy0, dx1, dy1] (normalized offsets)."""
    nominal = nominal_cell_boxes()
    target = torch.zeros(ROWS, COLS, 5)
    for i, slot in enumerate(shelf.slots):
        r, c = divmod(i, COLS)
        target[r, c, 0] = 1.0 if slot.occupied else 0.0
        if slot.occupied:
            x0, y0, x1, y1 = slot.bbox
            box_norm = torch.tensor([x0 / IMG_WIDTH, y0 / IMG_HEIGHT, x1 / IMG_WIDTH, y1 / IMG_HEIGHT])
            offset = (box_norm - nominal[r, c]) / 0.2
            target[r, c, 1:5] = offset
    return target


def batch_tensors(shelves: list[ShelfImage]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image_to_tensor(s) for s in shelves])
    targets = torch.stack([shelf_to_target(s) for s in shelves])
    return images, targets
