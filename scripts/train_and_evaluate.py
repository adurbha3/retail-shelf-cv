#!/usr/bin/env python
"""Generate synthetic shelf data, train the slot detector and price OCR model,
then report detection, OCR, and barcode-decode metrics on a held-out set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retailshelf.synthetic import generate_dataset
from retailshelf.data import batch_tensors
from retailshelf.detector import decode_predictions
from retailshelf.train import TrainConfig, train_detector, train_ocr
from retailshelf.ocr import generate_ocr_dataset, predict_digits
from retailshelf.barcode import decode_barcode, BarcodeDecodeError
from retailshelf.metrics import (
    evaluate_detections, aggregate_detection_metrics,
    ocr_exact_match_accuracy, ocr_char_accuracy, barcode_decode_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ocr-n-train", type=int, default=400,
                         help="OCR digit recognition needs more examples than the detector")
    parser.add_argument("--ocr-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Generating {args.n_train} training and {args.n_test} test shelf images...")
    train_shelves = generate_dataset(args.n_train, seed=args.seed)
    test_shelves = generate_dataset(args.n_test, seed=args.seed + 1)

    train_images, train_targets = batch_tensors(train_shelves)
    detector = train_detector(train_images, train_targets, TrainConfig(epochs=args.epochs))

    test_images, _ = batch_tensors(test_shelves)
    detector.eval()
    import torch
    with torch.no_grad():
        raw_preds = detector(test_images)

    per_image_metrics = []
    for shelf, raw in zip(test_shelves, raw_preds):
        detections = decode_predictions(raw)
        predicted_boxes = [d[3] for d in detections]
        gt_boxes = [s.bbox for s in shelf.slots if s.occupied]
        per_image_metrics.append(evaluate_detections(predicted_boxes, gt_boxes))
    detection_metrics = aggregate_detection_metrics(per_image_metrics)

    print("\nSlot detection (occupied vs. empty), on held-out shelves:")
    print(
        f"  precision={detection_metrics.precision:.3f} recall={detection_metrics.recall:.3f} "
        f"f1={detection_metrics.f1:.3f} mean_iou={detection_metrics.mean_iou:.3f}"
    )

    print(f"\nTraining price OCR model on {args.ocr_n_train} synthetic price-tag crops...")
    ocr_train_images, ocr_train_labels, _ = generate_ocr_dataset(args.ocr_n_train, seed=args.seed)
    ocr_model = train_ocr(ocr_train_images, ocr_train_labels, TrainConfig(epochs=args.ocr_epochs, batch_size=32))

    ocr_test_images, _, ocr_test_strings = generate_ocr_dataset(args.n_test, seed=args.seed + 2)
    ocr_predictions = predict_digits(ocr_model, ocr_test_images)
    exact = ocr_exact_match_accuracy(ocr_predictions, ocr_test_strings)
    char_acc = ocr_char_accuracy(ocr_predictions, ocr_test_strings)
    print(f"  OCR exact-match accuracy={exact:.3f} char-level accuracy={char_acc:.3f}")

    print("\nDecoding barcodes from the test shelf images (pure signal processing, no ML)...")
    decoded, ground_truth_skus = [], []
    for shelf in test_shelves:
        for slot in shelf.slots:
            if not slot.occupied:
                continue
            crop = shelf.image.crop(slot.barcode_bbox)
            try:
                decoded.append(decode_barcode(crop))
            except BarcodeDecodeError:
                decoded.append(None)
            ground_truth_skus.append(slot.sku)
    barcode_acc = barcode_decode_accuracy(decoded, ground_truth_skus)
    print(f"  barcode decode accuracy={barcode_acc:.3f} over {len(ground_truth_skus)} barcodes")


if __name__ == "__main__":
    main()
