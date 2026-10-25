from retailshelf.metrics import (
    iou, evaluate_detections, aggregate_detection_metrics,
    ocr_exact_match_accuracy, ocr_char_accuracy, barcode_decode_accuracy,
)


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    # intersection 5x5=25, union = 100+100-25=175
    assert abs(iou(a, b) - 25 / 175) < 1e-6


def test_evaluate_detections_perfect_match():
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
    metrics = evaluate_detections(boxes, boxes)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_evaluate_detections_false_positive_and_negative():
    predicted = [(0, 0, 10, 10), (100, 100, 110, 110)]  # second has no match
    ground_truth = [(0, 0, 10, 10), (50, 50, 60, 60)]  # second never predicted
    metrics = evaluate_detections(predicted, ground_truth)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1


def test_aggregate_detection_metrics_sums_counts():
    m1 = evaluate_detections([(0, 0, 10, 10)], [(0, 0, 10, 10)])
    m2 = evaluate_detections([], [(0, 0, 10, 10)])
    agg = aggregate_detection_metrics([m1, m2])
    assert agg.true_positives == 1
    assert agg.false_negatives == 1


def test_ocr_accuracy_metrics():
    predicted = ["0550", "2999", "0001"]
    ground_truth = ["0550", "2999", "0099"]
    assert ocr_exact_match_accuracy(predicted, ground_truth) == 2 / 3
    assert ocr_char_accuracy(predicted, ground_truth) == 10 / 12


def test_barcode_decode_accuracy_handles_none():
    decoded = [123, None, 456]
    ground_truth = [123, 789, 456]
    assert barcode_decode_accuracy(decoded, ground_truth) == 2 / 3
