"""A lightweight single-shot grid detector for shelf-slot occupancy.

Shelf slots sit on a known ROWS x COLS grid, so detection reduces to: for each
grid cell, predict whether it holds a product (objectness) and a bounding-box
offset refining the nominal cell box -- the same single-shot-per-cell idea
behind YOLO-style detectors, scaled down to this problem.
"""
from __future__ import annotations

import torch
from torch import nn

from retailshelf.synthetic import ROWS, COLS, IMG_WIDTH, IMG_HEIGHT


class SlotDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Reduce whatever spatial size the backbone produces down to the slot grid.
        self.pool_to_grid = nn.AdaptiveAvgPool2d((ROWS, COLS))
        self.head = nn.Conv2d(64, 5, kernel_size=1)  # [objectness, dx0, dy0, dx1, dy1] per cell

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, H, W) -> (batch, ROWS, COLS, 5)"""
        features = self.backbone(x)
        grid = self.pool_to_grid(features)
        out = self.head(grid)  # (batch, 5, ROWS, COLS)
        return out.permute(0, 2, 3, 1)  # (batch, ROWS, COLS, 5)


def nominal_cell_boxes() -> torch.Tensor:
    """The un-jittered grid cell boxes the model regresses offsets against, normalized to [0, 1]."""
    cell_w = IMG_WIDTH / COLS
    cell_h = IMG_HEIGHT / ROWS
    boxes = torch.zeros(ROWS, COLS, 4)
    for r in range(ROWS):
        for c in range(COLS):
            boxes[r, c] = torch.tensor([
                c * cell_w / IMG_WIDTH, r * cell_h / IMG_HEIGHT,
                (c + 1) * cell_w / IMG_WIDTH, (r + 1) * cell_h / IMG_HEIGHT,
            ])
    return boxes


def decode_predictions(raw: torch.Tensor, objectness_threshold: float = 0.5):
    """raw: (ROWS, COLS, 5) model output for one image -> list of (row, col, prob, bbox_xyxy_px)."""
    nominal = nominal_cell_boxes()
    probs = torch.sigmoid(raw[..., 0])
    boxes_norm = nominal + raw[..., 1:5] * 0.2  # small bounded offset around the nominal cell

    detections = []
    for r in range(ROWS):
        for c in range(COLS):
            prob = float(probs[r, c])
            if prob >= objectness_threshold:
                x0, y0, x1, y1 = boxes_norm[r, c].tolist()
                bbox_px = (x0 * IMG_WIDTH, y0 * IMG_HEIGHT, x1 * IMG_WIDTH, y1 * IMG_HEIGHT)
                detections.append((r, c, prob, bbox_px))
    return detections
