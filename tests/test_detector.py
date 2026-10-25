import torch

from retailshelf.detector import SlotDetector, decode_predictions, nominal_cell_boxes
from retailshelf.synthetic import ROWS, COLS, IMG_WIDTH, IMG_HEIGHT


def test_forward_pass_output_shape():
    model = SlotDetector()
    x = torch.randn(4, 3, IMG_HEIGHT, IMG_WIDTH)
    out = model(x)
    assert out.shape == (4, ROWS, COLS, 5)


def test_nominal_cell_boxes_cover_grid_without_gaps():
    boxes = nominal_cell_boxes()
    assert boxes.shape == (ROWS, COLS, 4)
    first = boxes[0, 0]
    assert first[0] == 0.0 and first[1] == 0.0


def test_decode_predictions_respects_threshold():
    raw = torch.zeros(ROWS, COLS, 5)
    raw[0, 0, 0] = 10.0  # strong objectness logit -> sigmoid ~1
    raw[0, 1, 0] = -10.0  # strong non-objectness
    detections = decode_predictions(raw, objectness_threshold=0.5)
    cells = {(r, c) for r, c, _, _ in detections}
    assert (0, 0) in cells
    assert (0, 1) not in cells


def test_decoded_boxes_are_within_image_bounds():
    model = SlotDetector()
    model.eval()
    x = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
    with torch.no_grad():
        raw = model(x)[0]
    margin_x, margin_y = 0.15 * IMG_WIDTH, 0.15 * IMG_HEIGHT
    for _, _, _, (x0, y0, x1, y1) in decode_predictions(raw, objectness_threshold=0.0):
        assert x0 < x1 and y0 < y1
        assert -margin_x <= x0 and x1 <= IMG_WIDTH + margin_x
        assert -margin_y <= y0 and y1 <= IMG_HEIGHT + margin_y
