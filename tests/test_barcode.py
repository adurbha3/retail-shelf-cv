from retailshelf.barcode import render_barcode, decode_barcode, encode_bits, BarcodeDecodeError
import pytest


def test_encode_bits_roundtrip_length():
    bits = encode_bits(123456)
    assert bits[:3] == [1, 0, 1]
    assert bits[-3:] == [1, 0, 1]


def test_render_and_decode_roundtrip():
    for value in [0, 1, 42, 123456, 999999]:
        img = render_barcode(value)
        assert decode_barcode(img) == value


def test_decode_is_robust_to_resizing():
    img = render_barcode(314159).resize((300, 50))
    assert decode_barcode(img) == 314159


def test_rejects_out_of_range_value():
    with pytest.raises(ValueError):
        render_barcode(10**6)
    with pytest.raises(ValueError):
        render_barcode(-1)


def test_decode_garbage_raises_decode_error():
    from PIL import Image
    blank = Image.new("L", (200, 40), color=128)
    with pytest.raises(BarcodeDecodeError):
        decode_barcode(blank)
