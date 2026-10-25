import torch

from retailshelf.ocr import (
    price_to_digits, render_price_crop, crop_to_tensor, PriceOCR,
    generate_ocr_dataset, predict_digits, CROP_WIDTH, CROP_HEIGHT, N_DIGITS,
)


def test_price_to_digits_formats_correctly():
    assert price_to_digits(5.5) == "0550"
    assert price_to_digits(0.99) == "0099"
    assert price_to_digits(29.99) == "2999"


def test_crop_to_tensor_shape():
    crop = render_price_crop(12.34, seed=1)
    tensor = crop_to_tensor(crop)
    assert tensor.shape == (1, CROP_HEIGHT, CROP_WIDTH)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_model_forward_pass_shape():
    model = PriceOCR()
    x = torch.randn(4, 1, CROP_HEIGHT, CROP_WIDTH)
    out = model(x)
    assert out.shape == (4, N_DIGITS, 10)


def test_generate_ocr_dataset_shapes_and_labels_match_strings():
    images, labels, strings = generate_ocr_dataset(6, seed=0)
    assert images.shape == (6, 1, CROP_HEIGHT, CROP_WIDTH)
    assert labels.shape == (6, N_DIGITS)
    for label_row, s in zip(labels.tolist(), strings):
        assert "".join(str(d) for d in label_row) == s


def test_predict_digits_returns_four_digit_strings():
    model = PriceOCR()
    images, _, _ = generate_ocr_dataset(3, seed=0)
    predictions = predict_digits(model, images)
    assert len(predictions) == 3
    assert all(len(p) == N_DIGITS and p.isdigit() for p in predictions)
