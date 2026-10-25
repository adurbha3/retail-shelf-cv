# Retail Shelf CV

**A computer vision pipeline for retail shelf monitoring: slot-occupancy detection (out-of-stock detection), price-tag OCR, and barcode decoding, with dataset generation and error-mining tooling built in.**

## Why this exists

My background so far (LiDAR point-cloud object detection at NC State's Trustworthy Computing for Autonomous Vehicles Lab, real-time perception on Purdue's IEEE Racing autonomous kart, and 1D CNN sensor-signal work at Purdue's Snowball Lab) has centered on turning raw, noisy sensor data into reliable, evaluated detections. This project applies that same discipline to 2D retail imagery: the exact kind of problem a shelf-scanning robot has to solve — is this slot occupied or out-of-stock, what does the price tag say, what SKU does this barcode encode — from images that are cluttered, blurry, and noisy by nature.

It ships with a seeded synthetic shelf-image generator (grid of product/empty slots, each with a price tag and barcode, rendered with Gaussian noise and blur) so the full pipeline runs with no proprietary retail dataset, while the module boundaries (image in, detections/text/SKU out) mirror what a real shelf-imagery system needs.

## Features

- **Synthetic shelf generator** (`retailshelf/synthetic.py`) — seeded shelf images on a configurable grid, each slot either a product (color, price tag, barcode) or an empty out-of-stock gap, with image noise and blur to mimic real camera conditions.
- **Slot detector** (`retailshelf/detector.py`) — a lightweight single-shot grid CNN (YOLO-style, one prediction per grid cell: objectness + bounding-box offset) that detects occupied vs. empty slots and refines each slot's bounding box.
- **Price-tag OCR** (`retailshelf/ocr.py`) — a small CNN that reads a "$DD.DD" price tag crop as four independent digit classifications, trained directly on synthetic price-tag crops.
- **Barcode decoding** (`retailshelf/barcode.py`) — a custom 1D barcode scheme (its own encode/decode, no external library) decoded via scanline thresholding and module sampling, the same signal-processing approach a real barcode reader uses.
- **Evaluation metrics** (`retailshelf/metrics.py`) — IoU-based precision/recall/F1/mean-IoU for detection (greedy matching, aggregated across a test set), exact-match and character-level accuracy for OCR, and decode accuracy for barcodes.
- **Error mining** (`scripts/mine_errors.py`) — runs the trained detector on a fresh test set, ranks images by number of mismatched slots, and saves annotated crops (ground truth vs. prediction) of the worst failures for manual review.
- **Tested** — 27 automated tests (`pytest`) covering the generator, detector forward pass and decoding, OCR model and digit formatting, metrics, and training loops — run automatically on push via GitHub Actions.

## Install

```bash
git clone https://github.com/adurbha3/retail-shelf-cv.git
cd retail-shelf-cv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs Python 3.9+.

## Usage

Train the slot detector and price OCR model on synthetic data, then report detection, OCR, and barcode metrics on a held-out set:

```bash
python scripts/train_and_evaluate.py --n-train 60 --n-test 20 --epochs 20
```

Render a preview of one synthetic shelf image with ground-truth slot/tag/barcode boxes:

```bash
python scripts/render_preview.py --out assets/shelf_preview.png
```

Mine the detector's worst failures on a fresh test set and save annotated crops:

```bash
python scripts/mine_errors.py --top-n 5 --out-dir assets/errors
```

## Project layout

```
retailshelf/
  synthetic.py    seeded shelf image + slot/price/barcode generator
  detector.py     grid-based slot-occupancy detector (SlotDetector)
  ocr.py          price-tag digit OCR (PriceOCR) + synthetic crop generator
  barcode.py      custom barcode encode/decode (pure signal processing)
  data.py         ShelfImage -> tensors and training targets
  train.py        training loops for the detector and OCR model
  metrics.py      detection IoU/precision/recall/F1, OCR accuracy, barcode accuracy
scripts/
  train_and_evaluate.py   CLI: train both models, report all metrics
  render_preview.py       CLI: save a ground-truth-annotated shelf image
  mine_errors.py          CLI: find and save the detector's worst failures
tests/            pytest suite covering every module above
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```
