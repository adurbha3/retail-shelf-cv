"""Price-tag OCR: a small CNN that reads a fixed-format "$DD.DD" price tag crop
as four independent digit classifications, plus a synthetic crop generator for
training it directly (without needing a full shelf scene).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont

CROP_WIDTH = 64
CROP_HEIGHT = 28
N_DIGITS = 4  # "$DD.DD" -> digits at positions 0,1 (dollars) and 2,3 (cents)


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def price_to_digits(price: float) -> str:
    """0.99 -> '0099' (dollars, then cents, no '$' or '.')."""
    text = f"{price:05.2f}"  # "$DD.DD" minus the leading '$'
    return text[0] + text[1] + text[3] + text[4]


def render_price_crop(price: float, noise_sigma: float = 5.0, seed: int | None = None) -> Image.Image:
    """Render one "$DD.DD" price tag crop, matching the style used in full shelf scenes."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (CROP_WIDTH, CROP_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f"${price:05.2f}", fill=0, font=_font())
    if noise_sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += rng.normal(0, noise_sigma, size=arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def crop_to_tensor(crop: Image.Image) -> torch.Tensor:
    resized = crop.convert("L").resize((CROP_WIDTH, CROP_HEIGHT))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


class PriceOCR(nn.Module):
    """Shared conv backbone, one linear classification head per digit position."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 7)),
        )
        self.flatten = nn.Flatten()
        self.heads = nn.ModuleList([nn.Linear(32 * 3 * 7, 10) for _ in range(N_DIGITS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, H, W) -> (batch, N_DIGITS, 10) logits."""
        features = self.flatten(self.backbone(x))
        return torch.stack([head(features) for head in self.heads], dim=1)


def generate_ocr_dataset(n: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rng = np.random.default_rng(seed)
    images, labels, strings = [], [], []
    for _ in range(n):
        price = round(float(rng.uniform(0.99, 29.99)), 2)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        crop = render_price_crop(price, seed=sub_seed)
        digit_str = price_to_digits(price)
        images.append(crop_to_tensor(crop))
        labels.append([int(d) for d in digit_str])
        strings.append(digit_str)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), strings


def predict_digits(model: PriceOCR, images: torch.Tensor) -> list[str]:
    model.eval()
    with torch.no_grad():
        logits = model(images)  # (batch, N_DIGITS, 10)
        preds = logits.argmax(dim=-1)  # (batch, N_DIGITS)
    return ["".join(str(d) for d in row.tolist()) for row in preds]
