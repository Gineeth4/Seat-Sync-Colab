"""
Smart Seat Sync — core occupancy engine
================================================================================
The web-framework-agnostic heart of the service. Given a frame, it:
  1. runs the detector to find *occupant* boxes (persons, by default),
  2. projects the fixed seat layout (normalised config) into pixel space,
  3. assigns occupants to seats by maximum IoU (core.seat_geometry), and
  4. returns a per-seat OCCUPIED/EMPTY verdict plus runtime telemetry.

Why seats come from config, not the detector
--------------------------------------------
Seat regions are static per camera, so trusting a per-frame seat *classification*
throws away that prior and flickers. Instead the seats are a one-time calibration
(configs/seat_layout.json in normalised coords), and the model only has to do the
hard, dynamic job: find people. This makes occupancy robust even when the seat
class is noisy — the failure mode we observed on the smoke model.

The geometry step (`assign`) is deliberately separated from model inference so it
can be unit-tested with synthetic detections and no GPU (see `_selftest`).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.seat_geometry import match_detections_to_seats, xywh_to_xyxy  # noqa: E402

_MB = 1024 * 1024
CLS_PERSON = 2  # default occupant class (matches configs/seat_data.yaml)


@dataclass
class SeatState:
    seat_id: str
    occupied: bool
    iou: float
    occupant_index: int   # index into the frame's occupant detections, -1 if none
    occupant_conf: float


@dataclass
class OccupancyResult:
    seats: list           # list[SeatState]
    n_occupants: int
    occupied_count: int
    seat_count: int
    latency_ms: float
    peak_vram_mb: float

    def to_dict(self) -> dict:
        return {
            "seats": [asdict(s) for s in self.seats],
            "n_occupants": self.n_occupants,
            "occupied_count": self.occupied_count,
            "seat_count": self.seat_count,
            "latency_ms": self.latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
        }


class OccupancyEngine:
    """Loads a YOLO model + a seat layout and turns frames into occupancy verdicts.

    Thread-safe: a lock serialises model inference so multiple HTTP/WebSocket
    callers cannot corrupt CUDA state on the single 4GB device.
    """

    def __init__(
        self,
        weights: str | Path,
        seat_layout_path: str | Path,
        device=None,
        conf: float = 0.25,
        iou_threshold: float = 0.30,
        imgsz: int = 640,
        occupant_classes: tuple[int, ...] = (CLS_PERSON,),
        _skip_model: bool = False,
    ) -> None:
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.occupant_classes = tuple(occupant_classes)
        self._lock = threading.Lock()

        # Telemetry
        self.frames_processed = 0
        self.last_latency_ms = 0.0
        self.ema_fps = 0.0
        self.peak_vram_mb = 0.0

        self._load_layout(seat_layout_path)

        if _skip_model:
            # Test / geometry-only mode: no torch, no GPU.
            self.torch = None
            self.model = None
            self.device = "cpu"
            self.use_half = False
            return

        import torch
        from ultralytics import YOLO

        self.torch = torch
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights}. Train first with m2_train.py.")
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.use_half = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.model = YOLO(str(weights))
        self._warmup()

    # -- setup ---------------------------------------------------------------
    def _load_layout(self, path: str | Path) -> None:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if spec.get("coord_space") != "normalized_xywh":
            raise ValueError("seat layout must declare coord_space 'normalized_xywh'")
        seats = spec["seats"]
        if not seats:
            raise ValueError("seat layout has no seats")
        self.layout_name = spec.get("name", "layout")
        self.seat_ids = [str(s["id"]) for s in seats]
        # (S, 4) normalised centre boxes.
        self._seat_norm = np.array(
            [[s["cx"], s["cy"], s["w"], s["h"]] for s in seats], dtype=np.float64)

    def _warmup(self) -> None:
        """One inference on a black frame so cuDNN autotuning is paid up-front."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(source=blank, imgsz=self.imgsz, conf=self.conf,
                           device=self.device, half=self.use_half, verbose=False)
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    # -- geometry (pure, testable) ------------------------------------------
    def seat_boxes_px(self, width: int, height: int) -> np.ndarray:
        """Project the normalised seat layout into pixel xyxy for a WxH frame."""
        scale = np.array([width, height, width, height], dtype=np.float64)
        return xywh_to_xyxy(self._seat_norm * scale)

    def assign(self, occupant_xyxy: np.ndarray, occupant_conf: np.ndarray,
               width: int, height: int) -> list[SeatState]:
        """Assign occupant boxes to configured seats by IoU. No model needed."""
        seat_xyxy = self.seat_boxes_px(width, height)
        assignments = match_detections_to_seats(seat_xyxy, occupant_xyxy, self.iou_threshold)
        states: list[SeatState] = []
        for a in assignments:
            conf = float(occupant_conf[a.detection_index]) if a.detection_index >= 0 else 0.0
            states.append(SeatState(
                seat_id=self.seat_ids[a.seat_index],
                occupied=a.occupied,
                iou=round(float(a.iou), 4),
                occupant_index=a.detection_index,
                occupant_conf=round(conf, 4),
            ))
        return states

    # -- inference -----------------------------------------------------------
    def infer(self, frame, conf: float | None = None) -> OccupancyResult:
        """Run detection + assignment on one frame (path str/Path or BGR ndarray).

        conf overrides the engine's configured gate for this call only (used by
        the /occupancy endpoints so the dashboard's confidence slider applies to
        seat detection too, not just the general /detect model).
        """
        if self.model is None:
            raise RuntimeError("engine built with _skip_model=True cannot run inference")

        with self._lock:
            cuda = self.torch.cuda.is_available()
            if cuda:
                self.torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = self.model.predict(source=frame, imgsz=self.imgsz,
                                       conf=conf if conf is not None else self.conf,
                                       device=self.device, half=self.use_half, verbose=False)[0]
            if cuda:
                self.torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) * 1000.0

            height, width = result.orig_shape  # ultralytics: (h, w)
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                occ_xyxy = np.empty((0, 4), dtype=np.float64)
                occ_conf = np.empty((0,), dtype=np.float64)
            else:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
                cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float64)
                mask = np.isin(cls, self.occupant_classes)
                occ_xyxy, occ_conf = xyxy[mask], conf[mask]

            states = self.assign(occ_xyxy, occ_conf, width, height)

            # telemetry (inside the lock so counters stay consistent)
            self.frames_processed += 1
            self.last_latency_ms = round(latency, 2)
            inst_fps = 1000.0 / latency if latency > 0 else 0.0
            self.ema_fps = inst_fps if self.ema_fps == 0 else 0.9 * self.ema_fps + 0.1 * inst_fps
            if cuda:
                self.peak_vram_mb = round(self.torch.cuda.max_memory_allocated() / _MB, 1)

        occupied = sum(1 for s in states if s.occupied)
        return OccupancyResult(
            seats=states,
            n_occupants=int(occ_xyxy.shape[0]),
            occupied_count=occupied,
            seat_count=len(states),
            latency_ms=round(latency, 2),
            peak_vram_mb=self.peak_vram_mb,
        )

    # -- introspection -------------------------------------------------------
    def layout(self) -> dict:
        return {
            "name": self.layout_name,
            "seats": [
                {"id": sid, "cx": float(b[0]), "cy": float(b[1]),
                 "w": float(b[2]), "h": float(b[3])}
                for sid, b in zip(self.seat_ids, self._seat_norm)
            ],
        }

    def stats(self) -> dict:
        return {
            "device": str(self.device),
            "half_precision": self.use_half,
            "layout": self.layout_name,
            "seat_count": len(self.seat_ids),
            "occupant_classes": list(self.occupant_classes),
            "conf": self.conf,
            "iou_threshold": self.iou_threshold,
            "frames_processed": self.frames_processed,
            "last_latency_ms": self.last_latency_ms,
            "ema_fps": round(self.ema_fps, 1),
            "peak_vram_mb": self.peak_vram_mb,
        }


# ----------------------------------------------------------------------------
# Self-test: validates the geometry/assignment wiring with NO model / GPU.
# ----------------------------------------------------------------------------
def _selftest() -> None:
    layout = _PROJECT_ROOT / "configs" / "seat_layout.json"
    eng = OccupancyEngine(weights="(unused)", seat_layout_path=layout, _skip_model=True)

    W = H = 640
    # Seat A1 centre (0.220, 0.290) -> pixel box; build an occupant that covers it.
    a1 = eng.seat_boxes_px(W, H)[0]  # xyxy of first seat
    occ = np.array([[a1[0] + 3, a1[1] + 3, a1[2] - 3, a1[3] - 3]], dtype=np.float64)
    conf = np.array([0.91])

    states = eng.assign(occ, conf, W, H)
    assert len(states) == 6, "expected 6 seats"
    assert states[0].seat_id == "A1" and states[0].occupied, "A1 must be occupied"
    assert states[0].iou > 0.7, f"A1 IoU should be high, got {states[0].iou}"
    assert sum(s.occupied for s in states) == 1, "only A1 should be occupied"

    # No occupants -> all empty.
    empty = eng.assign(np.empty((0, 4)), np.empty((0,)), W, H)
    assert all(not s.occupied for s in empty), "no occupants must yield all-empty"

    # Two occupants over A1 -> only one seat, one occupant assigned (one-to-one).
    two = np.array([occ[0], [a1[0] + 20, a1[1] + 20, a1[2] + 20, a1[3] + 20]], dtype=np.float64)
    st2 = eng.assign(two, np.array([0.9, 0.8]), W, H)
    assert sum(s.occupied for s in st2) == 1, "A1 must take exactly one occupant"

    print("[ OK ] occupancy_engine self-test passed (assignment wiring verified).")


if __name__ == "__main__":
    _selftest()
