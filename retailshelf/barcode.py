"""A compact 1D barcode: encode a numeric SKU as black/white bar modules, and
decode it back from a rendered image crop via scanline thresholding.

This is a custom scheme (not a real-world standard), designed to be simple to
render and to decode reliably from a noisy image, in the same way a real
barcode reader thresholds a scanline and samples module centers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

N_DIGITS = 6
BITS_PER_DIGIT = 4
START_STOP = [1, 0, 1]
MODULE_WIDTH_PX = 4
BAR_HEIGHT_PX = 36
QUIET_ZONE_PX = 12

MAX_VALUE = 10**N_DIGITS - 1


def _digit_to_bits(digit: int) -> list[int]:
    if not (0 <= digit <= 9):
        raise ValueError(f"digit out of range: {digit}")
    return [int(b) for b in format(digit, f"0{BITS_PER_DIGIT}b")]


def _bits_to_digit(bits: list[int]) -> int:
    return int("".join(str(b) for b in bits), 2)


def encode_bits(value: int) -> list[int]:
    """SKU integer -> full module bit sequence (including start/stop guards)."""
    if not (0 <= value <= MAX_VALUE):
        raise ValueError(f"value must be in [0, {MAX_VALUE}], got {value}")
    digits = [int(d) for d in str(value).zfill(N_DIGITS)]
    bits = list(START_STOP)
    for d in digits:
        bits.extend(_digit_to_bits(d))
    bits.extend(START_STOP)
    return bits


def barcode_width_px() -> int:
    n_modules = len(START_STOP) * 2 + N_DIGITS * BITS_PER_DIGIT
    return n_modules * MODULE_WIDTH_PX + 2 * QUIET_ZONE_PX


def render_barcode(value: int, height: int = BAR_HEIGHT_PX) -> Image.Image:
    """Render a SKU as a black/white barcode image (white background)."""
    bits = encode_bits(value)
    width = barcode_width_px()
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    x = QUIET_ZONE_PX
    for bit in bits:
        if bit == 1:
            draw.rectangle([x, 0, x + MODULE_WIDTH_PX - 1, height - 1], fill=0)
        x += MODULE_WIDTH_PX
    return img


class BarcodeDecodeError(ValueError):
    pass


def decode_barcode(crop: Image.Image) -> int:
    """Decode a (possibly noisy) cropped barcode image back into its SKU integer.

    Resizes the crop to the canonical render width, takes the middle scanline,
    thresholds it, and samples each module's center pixel.
    """
    width = barcode_width_px()
    resized = crop.convert("L").resize((width, BAR_HEIGHT_PX))
    arr = np.asarray(resized, dtype=np.float32)
    scanline = arr[arr.shape[0] // 2, :]

    threshold = (scanline.max() + scanline.min()) / 2.0
    n_modules = len(START_STOP) * 2 + N_DIGITS * BITS_PER_DIGIT
    bits = []
    for i in range(n_modules):
        center = QUIET_ZONE_PX + i * MODULE_WIDTH_PX + MODULE_WIDTH_PX // 2
        center = min(center, len(scanline) - 1)
        bits.append(1 if scanline[center] < threshold else 0)

    start, payload, stop = bits[:3], bits[3:-3], bits[-3:]
    if start != START_STOP or stop != START_STOP:
        raise BarcodeDecodeError("start/stop guard pattern not found")

    digits = []
    for i in range(N_DIGITS):
        digit_bits = payload[i * BITS_PER_DIGIT : (i + 1) * BITS_PER_DIGIT]
        digit = _bits_to_digit(digit_bits)
        if digit > 9:
            raise BarcodeDecodeError(f"invalid digit pattern decoded a value of {digit}")
        digits.append(digit)

    return int("".join(str(d) for d in digits))
