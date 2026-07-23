r"""
Smart Seat Sync — single-image detection viewer
================================================================================
Run the detector on ONE image, print what it found, and save an annotated copy
with the bounding boxes drawn so you can eyeball it.

Works with whichever weights you point it at:
  * your trained seat model (default) -> classes empty_seat / occupied_seat / person
  * stock COCO model (weights/yolov8n.pt) -> person, chair, etc. (for real photos)

Examples:
    py -3.10 scripts\detect_image.py C:\path\to\photo.jpg
    py -3.10 scripts\detect_image.py photo.jpg --weights weights\yolov8n.pt --conf 0.25 --show
    py -3.10 scripts\detect_image.py C:\Users\dhruv\datasets\seat_sync\images\test\test_00000.jpg --conf 0.05

--show opens the annotated result in your default image viewer (Windows).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}")
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect objects in a single image and visualize.")
    ap.add_argument("image", type=Path, help="Path to the image to run detection on.")
    ap.add_argument("--weights", type=Path,
                    default=_PROJECT_ROOT / "runs" / "smoke_test" / "weights" / "best.pt",
                    help="Model weights (default: your trained seat model).")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument("--show", action="store_true",
                    help="Open the annotated result in the default image viewer.")
    args = ap.parse_args(argv)

    if not args.image.is_file():
        _fatal(f"Image not found: {args.image}")
    if not args.weights.is_file():
        _fatal(f"Weights not found: {args.weights}. Train first, or use --weights weights\\yolov8n.pt")

    # Anchor CWD to project root so any stray Ultralytics writes stay writable.
    os.chdir(_PROJECT_ROOT)

    import cv2
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print("-" * 68)
    print(f" Detecting on: {args.image.name}")
    print(f" Weights     : {args.weights}  (device={device}, conf>={args.conf})")
    print("-" * 68)

    model = YOLO(str(args.weights))
    result = model.predict(source=str(args.image), imgsz=args.imgsz, conf=args.conf,
                           device=device, half=torch.cuda.is_available(), verbose=False)[0]

    names = model.names  # id -> class name (works for seat model OR COCO)
    boxes = result.boxes
    n = 0 if boxes is None else int(boxes.shape[0])

    if n == 0:
        print(" No detections above the confidence threshold.")
        print(" -> Try a lower --conf (e.g. 0.05), or a different --weights.")
    else:
        cls = boxes.cls.int().tolist()
        conf = boxes.conf.tolist()
        xyxy = boxes.xyxy.int().tolist()
        counts = Counter(names[c] for c in cls)
        print(f" {n} detection(s): " + ", ".join(f"{v}x {k}" for k, v in counts.items()))
        for c, cf, box in sorted(zip(cls, conf, xyxy), key=lambda t: -t[1]):
            x1, y1, x2, y2 = box
            print(f"   {names[c]:<15} conf={cf:.2f}  box=({x1},{y1})-({x2},{y2})")

    # Save the annotated image (boxes + labels drawn by Ultralytics).
    out_dir = _PROJECT_ROOT / "runs" / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"det_{args.image.stem}.jpg"
    annotated = result.plot()  # BGR ndarray with boxes/labels rendered
    cv2.imwrite(str(save_path), annotated)

    print("-" * 68)
    print(f" Annotated image saved: {save_path}")
    if args.show:
        if hasattr(os, "startfile"):
            os.startfile(str(save_path))  # noqa: S606 - open in default viewer (Windows)
        else:
            print(" (--show is Windows-only; open the file above manually.)")
    else:
        print(" Add --show to open it automatically.")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
