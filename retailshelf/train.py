"""Training loops for the slot detector and the price OCR model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from retailshelf.detector import SlotDetector
from retailshelf.ocr import PriceOCR


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3


def train_detector(images: torch.Tensor, targets: torch.Tensor, config: TrainConfig = TrainConfig()) -> SlotDetector:
    model = SlotDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()

    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for _ in range(config.epochs):
        for batch_images, batch_targets in loader:
            optimizer.zero_grad()
            pred = model(batch_images)  # (batch, ROWS, COLS, 5)

            objectness_loss = bce(pred[..., 0], batch_targets[..., 0])

            occupied_mask = batch_targets[..., 0] > 0.5
            if occupied_mask.any():
                bbox_loss = smooth_l1(pred[..., 1:5][occupied_mask], batch_targets[..., 1:5][occupied_mask])
            else:
                bbox_loss = torch.tensor(0.0)

            loss = objectness_loss + bbox_loss
            loss.backward()
            optimizer.step()

    return model


def train_ocr(images: torch.Tensor, labels: torch.Tensor, config: TrainConfig = TrainConfig()) -> PriceOCR:
    """labels: (n, N_DIGITS) integer digit labels."""
    model = PriceOCR()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for _ in range(config.epochs):
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_images)  # (batch, N_DIGITS, 10)
            loss = sum(
                criterion(logits[:, i, :], batch_labels[:, i]) for i in range(logits.shape[1])
            )
            loss.backward()
            optimizer.step()

    return model
