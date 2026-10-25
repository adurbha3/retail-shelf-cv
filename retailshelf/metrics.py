"""Evaluation metrics: slot-detection precision/recall/F1/mean IoU, OCR accuracy, barcode accuracy."""
from __future__ import annotations

from dataclasses import dataclass


def iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict:
        return {
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "mean_iou": self.mean_iou, "tp": self.true_positives,
            "fp": self.false_positives, "fn": self.false_negatives,
        }


def evaluate_detections(
    predicted_boxes: list, ground_truth_boxes: list, iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Greedy IoU matching between predicted and ground-truth occupied-slot boxes
    for a single image (same style as a precision/recall sweep over detector output)."""
    matched_gt = set()
    tp = 0
    ious = []
    for pred_box in predicted_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gt_box in enumerate(ground_truth_boxes):
            if i in matched_gt:
                continue
            score = iou(pred_box, gt_box)
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
            ious.append(best_iou)

    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return DetectionMetrics(
        precision=precision, recall=recall, f1=f1, mean_iou=mean_iou,
        true_positives=tp, false_positives=fp, false_negatives=fn,
    )


def aggregate_detection_metrics(per_image: list[DetectionMetrics]) -> DetectionMetrics:
    tp = sum(m.true_positives for m in per_image)
    fp = sum(m.false_positives for m in per_image)
    fn = sum(m.false_negatives for m in per_image)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    all_ious = [m.mean_iou for m in per_image if m.true_positives > 0]
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    return DetectionMetrics(
        precision=precision, recall=recall, f1=f1, mean_iou=mean_iou,
        true_positives=tp, false_positives=fp, false_negatives=fn,
    )


def ocr_exact_match_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / len(ground_truth)


def ocr_char_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    total_chars = sum(len(g) for g in ground_truth)
    if total_chars == 0:
        return 0.0
    correct_chars = 0
    for p, g in zip(predicted, ground_truth):
        for pc, gc in zip(p, g):
            if pc == gc:
                correct_chars += 1
    return correct_chars / total_chars


def barcode_decode_accuracy(decoded: list[int | None], ground_truth: list[int]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(1 for d, g in zip(decoded, ground_truth) if d == g)
    return correct / len(ground_truth)
