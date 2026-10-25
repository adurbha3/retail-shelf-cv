#!/usr/bin/env python
"""Error mining: train the detector, run it on a fresh test set, and save
annotated crops of its worst failures (false positives and false negatives)
for manual review -- the same workflow a CV team uses to find systematic
failure modes before the next data/model iteration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import ImageDraw

from retailshelf.synthetic import generate_dataset
from retailshelf.data import batch_tensors
from retailshelf.detector import decode_predictions, nominal_cell_boxes
from retailshelf.train import TrainConfig, train_detector
from retailshelf.synthetic import ROWS, COLS, IMG_WIDTH, IMG_HEIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)  # intentionally under-trained to surface errors
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="assets/errors")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_shelves = generate_dataset(args.n_train, seed=args.seed)
    train_images, train_targets = batch_tensors(train_shelves)
    detector = train_detector(train_images, train_targets, TrainConfig(epochs=args.epochs))
    detector.eval()

    test_shelves = generate_dataset(args.n_test, seed=args.seed + 100)
    test_images, _ = batch_tensors(test_shelves)
    with torch.no_grad():
        raw_preds = detector(test_images)

    nominal = nominal_cell_boxes()
    scored = []
    for idx, (shelf, raw) in enumerate(zip(test_shelves, raw_preds)):
        probs = torch.sigmoid(raw[..., 0])
        errors = 0
        mismatches = []
        for i, slot in enumerate(shelf.slots):
            r, c = divmod(i, COLS)
            predicted_occupied = probs[r, c].item() >= 0.5
            if predicted_occupied != slot.occupied:
                errors += 1
                kind = "false_positive" if predicted_occupied else "false_negative"
                mismatches.append((r, c, kind, probs[r, c].item()))
        scored.append((errors, idx, shelf, raw, mismatches))

    scored.sort(key=lambda t: t[0], reverse=True)
    worst = scored[: args.top_n]

    print(f"Scanned {args.n_test} test images; {sum(1 for s in scored if s[0] > 0)} had at least one error.")
    print(f"Saving the {len(worst)} worst to {out_dir}/\n")

    for rank, (n_errors, idx, shelf, raw, mismatches) in enumerate(worst):
        img = shelf.image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        detections = decode_predictions(raw)
        pred_boxes_by_cell = {(r, c): box for r, c, _, box in detections}

        for i, slot in enumerate(shelf.slots):
            r, c = divmod(i, COLS)
            draw.rectangle(slot.bbox, outline=(40, 180, 80) if slot.occupied else (120, 120, 120), width=1)
            if (r, c) in pred_boxes_by_cell:
                draw.rectangle(pred_boxes_by_cell[(r, c)], outline=(220, 40, 40), width=2)

        out_path = out_dir / f"error_{rank:02d}_image{idx}_errors{n_errors}.png"
        img.save(out_path)
        print(f"  [{rank}] image {idx}: {n_errors} mismatched slot(s) -> {out_path}")
        for r, c, kind, prob in mismatches:
            print(f"        cell ({r},{c}) {kind} (predicted prob={prob:.2f})")


if __name__ == "__main__":
    main()
