"""
Smart Seat Sync — synthetic dataset generator
================================================================================
Purpose
-------
You do not have labelled seat footage yet, but you need to prove the Milestone 2
training harness (loop + AMP + the VRAM monitor callbacks) actually runs on the
RTX 2050. This generates a small, *correctly labelled* YOLO dataset of stylised
"seats" and "people" so `m2_train.py` has something to consume end-to-end.

This is a SMOKE-TEST dataset, not a model that will generalise to real cameras.
Its only job is to make training exercise every code path with valid labels.

What it draws
-------------
A grid of seat rectangles on a neutral background. Each seat is randomly:
  * empty     -> one box, class 0 (empty_seat)
  * occupied  -> a seat box (class 1, occupied_seat) PLUS an overlapping person
                 blob (class 2, person) drawn on top, so the person/seat pair has
                 a realistic IoU — the exact geometry core/seat_geometry.py matches.

Label correctness
-----------------
Every box is emitted as `<class> <cx> <cy> <w> <h>`, all normalised to [0, 1] by
the image dimensions, clamped so no coordinate escapes the frame (Ultralytics
silently drops out-of-range labels, which would give a falsely "clean" run).

Layout written (matches configs/seat_data.yaml):
    <root>/images/train/*.jpg   <root>/labels/train/*.txt
    <root>/images/val/*.jpg     <root>/labels/val/*.txt

Run:
    py -3.10 scripts\gen_synthetic_dataset.py                 # 60 train / 20 val
    py -3.10 scripts\gen_synthetic_dataset.py --n-train 120 --n-val 30 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Class ids MUST match configs/seat_data.yaml.
CLS_EMPTY, CLS_OCCUPIED, CLS_PERSON = 0, 1, 2


def _xyxy_to_norm_xywh(x1: float, y1: float, x2: float, y2: float,
                       w_img: int, h_img: int) -> tuple[float, float, float, float]:
    """Pixel corner box -> normalised centre box, clamped into [0, 1].

    The clamp is not cosmetic: a person blob is intentionally allowed to spill a
    little past a seat, and a raw cx/w near the frame edge can round to >1.0,
    which Ultralytics drops. We clamp the *centre* and *size* after conversion so
    the emitted label is always in-range and non-degenerate.
    """
    cx = ((x1 + x2) / 2.0) / w_img
    cy = ((y1 + y2) / 2.0) / h_img
    bw = abs(x2 - x1) / w_img
    bh = abs(y2 - y1) / h_img
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-3), 1.0)
    bh = min(max(bh, 1e-3), 1.0)
    return cx, cy, bw, bh


def _make_one(rng: random.Random, imgsz: int) -> tuple[Image.Image, list[str]]:
    """Render one synthetic frame and return (image, label_lines)."""
    img = Image.new("RGB", (imgsz, imgsz), color=(210, 210, 205))
    draw = ImageDraw.Draw(img)
    labels: list[str] = []

    cols, rows = 3, 2
    margin = int(imgsz * 0.08)
    cell_w = (imgsz - 2 * margin) // cols
    cell_h = (imgsz - 2 * margin) // rows
    seat_w = int(cell_w * 0.62)
    seat_h = int(cell_h * 0.62)

    for r in range(rows):
        for c in range(cols):
            # Cell centre with a little positional jitter so frames are not identical.
            jx = rng.randint(-cell_w // 12, cell_w // 12)
            jy = rng.randint(-cell_h // 12, cell_h // 12)
            ccx = margin + c * cell_w + cell_w // 2 + jx
            ccy = margin + r * cell_h + cell_h // 2 + jy

            sx1, sy1 = ccx - seat_w // 2, ccy - seat_h // 2
            sx2, sy2 = ccx + seat_w // 2, ccy + seat_h // 2

            occupied = rng.random() < 0.5
            seat_cls = CLS_OCCUPIED if occupied else CLS_EMPTY
            seat_color = (90, 120, 200) if occupied else (150, 155, 160)
            draw.rectangle([sx1, sy1, sx2, sy2], fill=seat_color, outline=(40, 40, 40), width=2)
            labels.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                seat_cls, *_xyxy_to_norm_xywh(sx1, sy1, sx2, sy2, imgsz, imgsz)))

            if occupied:
                # Person blob: narrower than the seat, taller, sitting slightly high
                # and offset — this deliberately produces a partial (not perfect)
                # IoU with the seat, mirroring real occupancy geometry.
                pw = int(seat_w * 0.55)
                ph = int(seat_h * 1.15)
                pcx = ccx + rng.randint(-seat_w // 10, seat_w // 10)
                pcy = ccy - int(seat_h * 0.15)
                px1, py1 = pcx - pw // 2, pcy - ph // 2
                px2, py2 = pcx + pw // 2, pcy + ph // 2
                draw.ellipse([px1, py1, px2, py2], fill=(60, 60, 60))
                labels.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    CLS_PERSON, *_xyxy_to_norm_xywh(px1, py1, px2, py2, imgsz, imgsz)))

    return img, labels


def _write_split(root: Path, split: str, n: int, imgsz: int, rng: random.Random) -> int:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    boxes_total = 0
    for i in range(n):
        img, labels = _make_one(rng, imgsz)
        stem = f"{split}_{i:05d}"
        img.save(img_dir / f"{stem}.jpg", quality=90)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        boxes_total += len(labels)
    print(f"  {split:5s}: {n} images, {boxes_total} boxes -> {img_dir}")
    return boxes_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic YOLO seat dataset")
    parser.add_argument("--data", type=Path,
                        default=_PROJECT_ROOT / "configs" / "seat_data.yaml",
                        help="Dataset YAML whose 'path' is the output root.")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with args.data.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    root = Path(spec["path"])

    # Guard: the class map in the YAML must match the ids we emit, or the labels
    # are silently wrong. Fail loudly instead.
    names = spec["names"]
    expected = {CLS_EMPTY: "empty_seat", CLS_OCCUPIED: "occupied_seat", CLS_PERSON: "person"}
    for cid, cname in expected.items():
        if names.get(cid) != cname:
            print(f"[FATAL] class id {cid} in {args.data.name} is '{names.get(cid)}', "
                  f"expected '{cname}'. Fix the YAML or this generator before proceeding.")
            return 1

    print("-" * 68)
    print(f" Generating synthetic seat dataset -> {root}")
    print(f" seed={args.seed}  imgsz={args.imgsz}  (smoke-test data, not production)")
    print("-" * 68)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)  # keep any downstream numpy consumers deterministic too

    total = 0
    total += _write_split(root, "train", args.n_train, args.imgsz, rng)
    total += _write_split(root, "val", args.n_val, args.imgsz, rng)
    # A tiny test split lets `m2_train.py ... ` and later mAP eval find images.
    total += _write_split(root, "test", max(4, args.n_val // 2), args.imgsz, rng)

    print("-" * 68)
    print(f" Done. {total} boxes total across 3 classes.")
    print(" Next: py -3.10 scripts\\m2_train.py --profile safe --epochs 3")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    # Uniform Windows guard (see m2_train.py). This script spawns no workers, but
    # we keep the idiom consistent across every entrypoint.
    raise SystemExit(main())
