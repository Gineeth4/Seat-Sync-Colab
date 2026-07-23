"""
Smart Seat Sync — Milestone 3 (bridge)
Inference -> IoU seat-occupancy, end-to-end on hardware
================================================================================

Purpose
-------
Close the loop the training milestone left open: take a trained detector, run it
on real frames, and drive the detections through core/seat_geometry.py to produce
a per-seat OCCUPIED/EMPTY verdict. This is the exact function the FastAPI endpoint
will call in the next milestone, so proving it here (with a runtime self-audit)
de-risks the service layer.

Two independent occupancy signals, deliberately cross-checked
------------------------------------------------------------
The detector emits three classes: empty_seat(0), occupied_seat(1), person(2).
That already gives a *direct* occupancy label per seat. So why also compute IoU?

  * MODEL signal   — the seat's own predicted class (0 vs 1). Cheap, but a single
                     frame's classification flickers and has no notion of *which*
                     person occupies *which* seat.
  * GEOMETRY signal — assign each detected person to a seat region by maximum IoU
                     (core.seat_geometry.match_detections_to_seats). Scale-invariant,
                     deterministic, and in the real system the seat regions can come
                     from a fixed floor-plan config instead of the detector, which is
                     far more stable than trusting per-frame seat classification.

We report BOTH and flag disagreements. Agreement across frames is a live quality
signal: a seat the model calls "occupied" but that no person box covers is either
a classification error or a mislabelled seat region — exactly what we want surfaced
before it reaches the dashboard.

Run:
    py -3.10 scripts\m3_infer_occupancy.py                          # runs on the test split
    py -3.10 scripts\m3_infer_occupancy.py --source path\to\img.jpg
    py -3.10 scripts\m3_infer_occupancy.py --weights runs\smoke_test\weights\best.pt --save
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make `core` importable regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.seat_geometry import match_detections_to_seats  # noqa: E402

try:
    import torch
    import yaml
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] Missing dependency: {exc}. Run: py -3.10 -m pip install -r requirements.txt")
    raise SystemExit(2) from exc

_MB = 1024 * 1024

# Class ids — must match configs/seat_data.yaml.
CLS_EMPTY, CLS_OCCUPIED, CLS_PERSON = 0, 1, 2
_SEAT_CLASSES = (CLS_EMPTY, CLS_OCCUPIED)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# Detection parsing
# ----------------------------------------------------------------------------
def _split_detections(result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one Ultralytics result into seat vs person arrays.

    Returns:
        seat_xyxy   (S, 4)  pixel corners of seat detections (classes 0/1)
        seat_cls    (S,)    their predicted class id
        seat_conf   (S,)    their confidence
        person_xyxy (P, 4)  pixel corners of person detections (class 2)
        person_conf (P,)    their confidence
    """
    boxes = result.boxes
    if boxes is None or boxes.shape[0] == 0:
        z4 = np.empty((0, 4), dtype=np.float64)
        z1 = np.empty((0,), dtype=np.float64)
        return z4, z1, z1, z4, z1

    # .cpu().numpy() once; these are small per-frame arrays.
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float64)

    seat_mask = np.isin(cls, _SEAT_CLASSES)
    person_mask = cls == CLS_PERSON
    return (
        xyxy[seat_mask], cls[seat_mask], conf[seat_mask],
        xyxy[person_mask], conf[person_mask],
    )


# ----------------------------------------------------------------------------
# Per-frame occupancy
# ----------------------------------------------------------------------------
class FrameReport:
    __slots__ = ("path", "n_seats", "geom_occupied", "model_occupied",
                 "agreements", "latency_ms", "rows")

    def __init__(self, path: str) -> None:
        self.path = path
        self.n_seats = 0
        self.geom_occupied = 0
        self.model_occupied = 0
        self.agreements = 0
        self.latency_ms = 0.0
        self.rows: list[str] = []


def _analyse_frame(seat_xyxy, seat_cls, seat_conf, person_xyxy, person_conf,
                   iou_threshold: float, path: str) -> FrameReport:
    """Run the geometry assignment and reconcile it with the model's own labels."""
    rep = FrameReport(path)
    rep.n_seats = int(seat_xyxy.shape[0])

    assignments = match_detections_to_seats(seat_xyxy, person_xyxy, iou_threshold)
    for a in assignments:
        model_occ = bool(seat_cls[a.seat_index] == CLS_OCCUPIED)
        geom_occ = a.occupied
        agree = model_occ == geom_occ
        rep.geom_occupied += int(geom_occ)
        rep.model_occupied += int(model_occ)
        rep.agreements += int(agree)

        who = f"person#{a.detection_index}" if a.detection_index >= 0 else "—"
        flag = "OK" if agree else "DISAGREE"
        rep.rows.append(
            f"    seat{a.seat_index}: geom={'OCC ' if geom_occ else 'EMPTY'} "
            f"(iou={a.iou:.2f}, {who}) | model={'OCC ' if model_occ else 'EMPTY'} "
            f"(conf={seat_conf[a.seat_index]:.2f}) [{flag}]"
        )
    return rep


# ----------------------------------------------------------------------------
# Overlay rendering (visual proof)
# ----------------------------------------------------------------------------
def _draw_overlay(img_path: Path, seat_xyxy, seat_cls, person_xyxy,
                  assignments, out_path: Path) -> None:
    """Draw seat boxes coloured by GEOMETRY occupancy + person boxes, save to disk."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for p in person_xyxy:
        draw.rectangle(list(p), outline=(40, 120, 255), width=2)

    for a in assignments:
        box = seat_xyxy[a.seat_index]
        color = (30, 200, 60) if a.occupied else (150, 150, 150)
        draw.rectangle(list(box), outline=color, width=3)
        label = f"s{a.seat_index} {'OCC' if a.occupied else 'EMP'} {a.iou:.2f}"
        draw.text((box[0] + 3, box[1] + 3), label, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


# ----------------------------------------------------------------------------
# Source resolution
# ----------------------------------------------------------------------------
def _resolve_sources(source: str | None, data_yaml: Path) -> list[Path]:
    """A single image, a directory of images, or (default) the dataset test split."""
    if source:
        p = Path(source)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
        raise SystemExit(f"[FATAL] --source not found: {p}")

    with data_yaml.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    root = Path(spec["path"])
    test_dir = root / spec.get("test", "images/test")
    if not test_dir.is_dir():
        raise SystemExit(f"[FATAL] No --source given and test split missing: {test_dir}")
    imgs = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise SystemExit(f"[FATAL] Test split has no images: {test_dir}")
    return imgs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference -> IoU seat-occupancy bridge")
    parser.add_argument("--weights", type=Path,
                        default=_PROJECT_ROOT / "runs" / "smoke_test" / "weights" / "best.pt")
    parser.add_argument("--data", type=Path,
                        default=_PROJECT_ROOT / "configs" / "seat_data.yaml")
    parser.add_argument("--source", default=None,
                        help="Image file or dir. Default: the dataset test split.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence gate.")
    parser.add_argument("--iou-threshold", type=float, default=0.30,
                        help="Min person/seat IoU to count a seat occupied.")
    parser.add_argument("--fps-budget", type=float, default=15.0,
                        help="Frames slower than 1/budget are flagged as dropped.")
    parser.add_argument("--save", action="store_true",
                        help="Write annotated overlays to runs/occupancy_smoke/.")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"[FATAL] Weights not found: {args.weights}\n"
              "        Train first, e.g. py -3.10 scripts\\m2_train.py --profile safe --epochs 3")
        return 1

    sources = _resolve_sources(args.source, args.data)
    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()
    budget_ms = 1000.0 / args.fps_budget

    print("-" * 68)
    print(f" Occupancy bridge | weights={args.weights.name} | device={device} "
          f"| {len(sources)} frame(s)")
    print(f" conf>={args.conf} | seat-occupancy IoU>={args.iou_threshold} "
          f"| fps budget={args.fps_budget:.0f}")
    print("-" * 68)

    from ultralytics import YOLO
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(args.weights))

    # Warm up so cuDNN autotuning is not billed to the first frame's latency.
    model.predict(source=str(sources[0]), imgsz=args.imgsz, conf=args.conf,
                  device=device, half=use_half, verbose=False)

    out_dir = _PROJECT_ROOT / "runs" / "occupancy_smoke"
    reports: list[FrameReport] = []
    dropped = 0

    for src in sources:
        t0 = time.perf_counter()
        result = model.predict(source=str(src), imgsz=args.imgsz, conf=args.conf,
                               device=device, half=use_half, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        seat_xyxy, seat_cls, seat_conf, person_xyxy, person_conf = _split_detections(result)
        rep = _analyse_frame(seat_xyxy, seat_cls, seat_conf, person_xyxy, person_conf,
                             args.iou_threshold, src.name)
        rep.latency_ms = latency_ms
        reports.append(rep)
        if latency_ms > budget_ms:
            dropped += 1

        agree_str = f"{rep.agreements}/{rep.n_seats}" if rep.n_seats else "n/a"
        print(f"  {src.name:22s} | seats={rep.n_seats} persons={person_xyxy.shape[0]} "
              f"| occ geom={rep.geom_occupied} model={rep.model_occupied} "
              f"| agree={agree_str} | {latency_ms:5.1f} ms")
        for row in rep.rows:
            print(row)

        if args.save:
            assignments = match_detections_to_seats(seat_xyxy, person_xyxy, args.iou_threshold)
            _draw_overlay(src, seat_xyxy, seat_cls, person_xyxy, assignments,
                          out_dir / f"occ_{src.stem}.jpg")

    # ---- Aggregate self-audit ----
    n = len(reports)
    total_seats = sum(r.n_seats for r in reports)
    total_agree = sum(r.agreements for r in reports)
    lat = sorted(r.latency_ms for r in reports)
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    mean_fps = 1000.0 / (sum(lat) / n) if n else 0.0
    peak_vram = (torch.cuda.max_memory_allocated() / _MB) if torch.cuda.is_available() else 0.0
    agree_rate = (total_agree / total_seats * 100.0) if total_seats else 0.0

    print("-" * 68)
    print(f"  frames                : {n}")
    print(f"  seats analysed        : {total_seats}")
    print(f"  geom/model agreement  : {total_agree}/{total_seats} ({agree_rate:.1f}%)")
    print(f"  latency p50 / p95     : {p50:.1f} / {p95:.1f} ms")
    print(f"  throughput            : {mean_fps:.1f} FPS")
    print(f"  frames over budget    : {dropped}/{n}")
    if torch.cuda.is_available():
        print(f"  peak inference VRAM   : {peak_vram:.0f} MB "
              f"({peak_vram / (torch.cuda.get_device_properties(0).total_memory / _MB):.0%} of card)")
    if args.save:
        print(f"  overlays written to   : {out_dir}")
    print("-" * 68)
    print(f"  Status: {'OK' if dropped == 0 else f'{dropped} frame(s) over budget'}")
    return 0


if __name__ == "__main__":
    # Uniform Windows multiprocessing guard (see m2_train.py). Ultralytics predict
    # can spawn workers on some sources; keep the guard consistent everywhere.
    raise SystemExit(main())
