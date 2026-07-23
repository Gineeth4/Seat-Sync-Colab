"""
Smart Seat Sync — FastAPI service (Milestone 4)
================================================================================
Wraps core.occupancy_engine behind a local HTTP + WebSocket API for the React
dashboard.

Endpoints
---------
  GET  /health              liveness probe
  GET  /status              engine telemetry (device, fps, VRAM, frames, clients)
  GET  /seats               the configured seat layout (normalised coords)
  POST /occupancy           multipart image upload -> per-seat occupancy JSON
  WS   /ws/occupancy        server pushes live occupancy for a frame source

Streaming model
---------------
A single background producer runs the inference loop over a frame source and
broadcasts each result to all connected dashboards — one GPU, many viewers. The
producer starts when the first client connects and stops when the last leaves, so
an idle server never touches the GPU.

Config via environment (all optional; sane local defaults):
  SEAT_WEIGHTS           path to best.pt          (default runs/smoke_test/...)
  SEAT_LAYOUT            path to seat_layout.json
  SEAT_SOURCE           dir of images | video file | webcam index (default: test split)
  SEAT_CONF             detector confidence gate  (default 0.25)
  SEAT_IOU              occupancy IoU threshold   (default 0.30)
  SEAT_OCCUPANT_CLASSES comma ids counted as occupant (default "2" = person)
  SEAT_STREAM_FPS       producer target FPS       (default 4)
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import os
import sys
import threading
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from core.occupancy_engine import OccupancyEngine  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
def _default_source() -> str:
    """Fall back to the dataset test split so the stream works out of the box."""
    data_yaml = _PROJECT_ROOT / "configs" / "seat_data.yaml"
    try:
        spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        return str(Path(spec["path"]) / spec.get("test", "images/test"))
    except Exception:
        return str(_PROJECT_ROOT / "runs")


class Config:
    weights = os.environ.get(
        "SEAT_WEIGHTS", str(_PROJECT_ROOT / "runs" / "smoke_test" / "weights" / "best.pt"))
    layout = os.environ.get("SEAT_LAYOUT", str(_PROJECT_ROOT / "configs" / "seat_layout.json"))
    source = os.environ.get("SEAT_SOURCE") or _default_source()
    conf = float(os.environ.get("SEAT_CONF", "0.25"))
    iou = float(os.environ.get("SEAT_IOU", "0.30"))
    occupant_classes = tuple(
        int(x) for x in os.environ.get("SEAT_OCCUPANT_CLASSES", "2").split(",") if x.strip())
    stream_fps = float(os.environ.get("SEAT_STREAM_FPS", "4"))
    # The /detect endpoint uses a general-purpose model so real photos of people
    # actually get detected. Defaults to the stock COCO yolov8n (person, chair, ...).
    detect_weights = os.environ.get(
        "SEAT_DETECT_WEIGHTS", str(_PROJECT_ROOT / "weights" / "yolov8n.pt"))
    detect_imgsz = int(os.environ.get("SEAT_DETECT_IMGSZ", "640"))


# ----------------------------------------------------------------------------
# Frame source
# ----------------------------------------------------------------------------
def frame_source(source: str):
    """Yield (label, frame) forever. frame is a path str (images) or BGR ndarray (video).

    - directory  -> cycles its image files endlessly
    - video file -> loops the video
    - digit      -> live webcam (does not loop; ends if the device closes)
    """
    p = Path(source)
    if p.is_dir():
        imgs = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
        if not imgs:
            raise RuntimeError(f"no images in source dir {p}")
        for path in itertools.cycle(imgs):
            yield Path(path).name, path
        return

    is_webcam = source.isdigit()
    is_video = p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    if not (is_webcam or is_video):
        raise RuntimeError(f"unsupported SEAT_SOURCE: {source}")

    cap_arg = int(source) if is_webcam else source
    idx = 0
    while True:
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source {source}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            yield f"frame_{idx:06d}", frame
        cap.release()
        if is_webcam:  # a closed webcam should not spin forever
            break


# ----------------------------------------------------------------------------
# WebSocket connection manager + producer
# ----------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
engine: OccupancyEngine | None = None
_producer_task: asyncio.Task | None = None


async def _producer_loop() -> None:
    """Pull frames, run the engine off the event loop, broadcast, pace to target FPS."""
    interval = 1.0 / max(Config.stream_fps, 0.1)
    for label, frame in frame_source(Config.source):
        if not manager.active:
            break  # last client left -> stop touching the GPU
        result = await run_in_threadpool(engine.infer, frame)
        payload = {"type": "occupancy", "frame": label, **result.to_dict()}
        await manager.broadcast(payload)
        await asyncio.sleep(interval)


async def _ensure_producer() -> None:
    global _producer_task
    if _producer_task is None or _producer_task.done():
        _producer_task = asyncio.create_task(_producer_loop())


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = OccupancyEngine(
        weights=Config.weights,
        seat_layout_path=Config.layout,
        conf=Config.conf,
        iou_threshold=Config.iou,
        occupant_classes=Config.occupant_classes,
    )
    print(f"[service] engine ready: {engine.stats()}")
    print(f"[service] stream source: {Config.source}")
    yield
    if _producer_task is not None:
        _producer_task.cancel()


app = FastAPI(title="Smart Seat Sync", version="1.0.0", lifespan=lifespan)

# React dev servers (Vite 5173, CRA 3000). Local-only service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine_ready": engine is not None}


@app.get("/status")
async def status() -> dict:
    if engine is None:
        raise HTTPException(503, "engine not initialised")
    return {**engine.stats(), "source": Config.source,
            "stream_fps": Config.stream_fps, "clients": len(manager.active)}


@app.get("/seats")
async def seats() -> dict:
    if engine is None:
        raise HTTPException(503, "engine not initialised")
    return engine.layout()


@app.post("/occupancy")
async def occupancy(file: UploadFile) -> dict:
    if engine is None:
        raise HTTPException(503, "engine not initialised")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode uploaded image")
    result = await run_in_threadpool(engine.infer, img)
    return result.to_dict()


# ----------------------------------------------------------------------------
# General object detection on an uploaded image (/detect)
# ----------------------------------------------------------------------------
_detector = None
_detector_lock = threading.Lock()


def _get_detector():
    """Lazy-load the general detector on first use so idle servers don't hold it."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from ultralytics import YOLO
                m = YOLO(Config.detect_weights)
                print(f"[service] detector loaded: {Config.detect_weights}")
                _detector = m
    return _detector


def run_detection(img_bgr: np.ndarray, conf: float) -> dict:
    """Detect objects, return the annotated image (base64 JPEG) + a detection list."""
    model = _get_detector()
    device = 0 if torch.cuda.is_available() else "cpu"
    result = model.predict(source=img_bgr, imgsz=Config.detect_imgsz, conf=conf,
                           device=device, half=torch.cuda.is_available(), verbose=False)[0]
    names = model.names
    detections = []
    boxes = result.boxes
    if boxes is not None and boxes.shape[0]:
        cls = boxes.cls.int().tolist()
        cf = boxes.conf.tolist()
        xyxy = boxes.xyxy.int().tolist()
        for c, p, b in zip(cls, cf, xyxy):
            detections.append({"cls": names[c], "conf": round(float(p), 3), "box": b})

    annotated = result.plot()  # BGR ndarray with boxes + labels drawn
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    h, w = result.orig_shape
    counts = dict(Counter(d["cls"] for d in detections))
    return {
        "width": int(w),
        "height": int(h),
        "n_detections": len(detections),
        "counts": counts,
        "person_count": counts.get("person", 0),
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{b64}",
    }


@app.post("/detect")
async def detect(file: UploadFile, conf: float = 0.25) -> dict:
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode uploaded image")
    return await run_in_threadpool(run_detection, img, conf)


@app.websocket("/ws/occupancy")
async def ws_occupancy(ws: WebSocket) -> None:
    await manager.connect(ws)
    await _ensure_producer()
    try:
        # Park until the client disconnects; we don't require inbound messages,
        # but reading lets us detect the close promptly.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
