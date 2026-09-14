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
  POST /occupancy/video      multipart video upload -> sampled-frame per-seat occupancy timeline
  POST /detect               multipart image upload -> general (COCO) detection JSON
  POST /detect/video         multipart video upload -> sampled-frame detection timeline
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
import tempfile
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
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_TMP_UPLOAD_DIR = _PROJECT_ROOT / "runs" / "tmp"
_TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    # Library-specific object filtering: 0 (person), 56 (chair), 60 (table)
    detect_classes = tuple(
        int(x) for x in os.environ.get("SEAT_DETECT_CLASSES", "0,56,60").split(",") if x.strip().isdigit()
    )
    # /detect/video samples this many frames spread evenly across the upload
    # instead of running every frame, so a multi-minute clip still returns in
    # seconds on a 4GB card. Overridable per-request up to _VIDEO_MAX_FRAMES_CAP.
    detect_video_max_frames = int(os.environ.get("SEAT_DETECT_VIDEO_MAX_FRAMES", "24"))
    # Annotated frame thumbnails are downscaled to this width so a 24-frame
    # timeline response stays a few MB instead of tens of MB.
    detect_video_thumb_width = int(os.environ.get("SEAT_DETECT_VIDEO_THUMB_WIDTH", "640"))
    detect_video_max_upload_mb = float(os.environ.get("SEAT_DETECT_VIDEO_MAX_MB", "200"))


_VIDEO_MAX_FRAMES_CAP = 60  # hard ceiling regardless of the requested max_frames


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


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR numpy ndarray.
    Supports JPG, PNG, WEBP, AVIF, HEIC, BMP, and other formats.
    """
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from io import BytesIO
        from PIL import Image
        import pillow_avif  # noqa: F401
        import pillow_heif  # noqa: F401
        pil_img = Image.open(BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@app.post("/occupancy")
async def occupancy(file: UploadFile, conf: float | None = None) -> dict:
    if engine is None:
        raise HTTPException(503, "engine not initialised")
    data = await file.read()
    img = decode_image(data)
    if img is None:
        raise HTTPException(400, "could not decode uploaded image")
    result = await run_in_threadpool(engine.infer, img, conf)
    return result.to_dict()


def run_occupancy_video(video_path: str, conf: float | None, max_frames: int) -> dict:
    """Sample frames evenly across a video and run the seat engine on each,
    mirroring run_detection_video's sampling so a shared frame index lines up
    the two timelines in the UI."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open uploaded video (unsupported codec/container?)")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("could not determine frame count for this video; try re-exporting as .mp4")

        indices = _sample_frame_indices(total_frames, max_frames)
        timeline = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            result = engine.infer(frame, conf)
            timeline.append({
                "frame_index": idx,
                "timestamp_sec": round(idx / fps, 2) if fps > 0 else None,
                **result.to_dict(),
            })
    finally:
        cap.release()

    occupied_counts = [f["occupied_count"] for f in timeline]
    return {
        "fps": round(fps, 2) if fps > 0 else None,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else None,
        "sampled_frames": len(timeline),
        "seat_count": len(engine.seat_ids),
        "peak_occupied_count": max(occupied_counts) if occupied_counts else 0,
        "avg_occupied_count": round(sum(occupied_counts) / len(occupied_counts), 2) if occupied_counts else 0,
        "timeline": timeline,
    }


@app.post("/occupancy/video")
async def occupancy_video(file: UploadFile, conf: float | None = None, max_frames: int | None = None) -> dict:
    if engine is None:
        raise HTTPException(503, "engine not initialised")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video format '{ext or '(none)'}'. Use one of {sorted(_VIDEO_EXTS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    max_mb = Config.detect_video_max_upload_mb
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"video too large ({len(data) / 1e6:.1f} MB); limit is {max_mb:.0f} MB")

    frame_cap = max_frames if max_frames is not None else Config.detect_video_max_frames
    frame_cap = max(1, min(frame_cap, _VIDEO_MAX_FRAMES_CAP))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=str(_TMP_UPLOAD_DIR)) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return await run_in_threadpool(run_occupancy_video, tmp_path, conf, frame_cap)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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


def _parse_classes_param(classes: str | None) -> list[int] | None:
    """'all' -> [] (no filter), 'a,b,c' -> [int, ...], None -> None (use Config default)."""
    if classes is None:
        return None
    if classes.strip().lower() == "all":
        return []
    return [int(x) for x in classes.split(",") if x.strip().isdigit()]


def _resize_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max_width <= 0 or w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def run_detection(img_bgr: np.ndarray, conf: float, classes: list[int] | tuple[int, ...] | None = None,
                   thumb_width: int | None = None) -> dict:
    """Detect objects, return the annotated image (base64 JPEG) + a detection list.

    thumb_width caps the annotated PREVIEW's width only — detection still runs
    at Config.detect_imgsz, so accuracy is unaffected; it just shrinks the
    payload (used by /detect/video, which returns many frames per response).
    """
    model = _get_detector()
    device = 0 if torch.cuda.is_available() else "cpu"
    target_classes = list(classes if classes is not None else Config.detect_classes)
    predict_kwargs = {}
    if target_classes and target_classes != ["all"]:
        predict_kwargs["classes"] = target_classes

    result = model.predict(source=img_bgr, imgsz=Config.detect_imgsz, conf=conf,
                           device=device, half=torch.cuda.is_available(), verbose=False,
                           **predict_kwargs)[0]
    names = model.names
    detections = []
    boxes = result.boxes
    if boxes is not None and boxes.shape[0]:
        cls = boxes.cls.int().tolist()
        cf = boxes.conf.tolist()
        xyxy = boxes.xyxy.int().tolist()
        for c, p, b in zip(cls, cf, xyxy):
            cls_name = names[c]
            if cls_name == "dining table":
                cls_name = "table"
            detections.append({"cls": cls_name, "conf": round(float(p), 3), "box": b})

    annotated = result.plot()  # BGR ndarray with boxes + labels drawn
    if thumb_width:
        annotated = _resize_max_width(annotated, thumb_width)
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
async def detect(file: UploadFile, conf: float = 0.25, classes: str | None = None) -> dict:
    data = await file.read()
    img = decode_image(data)
    if img is None:
        raise HTTPException(400, "could not decode uploaded image")
    return await run_in_threadpool(run_detection, img, conf, _parse_classes_param(classes))


# ----------------------------------------------------------------------------
# General object detection on an uploaded video (/detect/video)
# ----------------------------------------------------------------------------
def _sample_frame_indices(total_frames: int, target_count: int) -> list[int]:
    """Evenly spaced frame indices covering the whole clip, not just its start."""
    if total_frames <= 0 or target_count <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))
    step = total_frames / target_count
    return [int(i * step) for i in range(target_count)]


def run_detection_video(video_path: str, conf: float, classes: list[int] | tuple[int, ...] | None,
                        max_frames: int) -> dict:
    """Sample frames evenly across a video file, run detection on each, and
    return a timeline plus aggregate stats. Raises ValueError on a video the
    file can't be opened or whose frame count can't be determined."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open uploaded video (unsupported codec/container?)")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("could not determine frame count for this video; try re-exporting as .mp4")

        indices = _sample_frame_indices(total_frames, max_frames)
        timeline = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            det = run_detection(frame, conf, classes, thumb_width=Config.detect_video_thumb_width)
            timeline.append({
                "frame_index": idx,
                "timestamp_sec": round(idx / fps, 2) if fps > 0 else None,
                **det,
            })
    finally:
        cap.release()

    person_counts = [f["person_count"] for f in timeline]
    return {
        "fps": round(fps, 2) if fps > 0 else None,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else None,
        "sampled_frames": len(timeline),
        "peak_person_count": max(person_counts) if person_counts else 0,
        "avg_person_count": round(sum(person_counts) / len(person_counts), 2) if person_counts else 0,
        "timeline": timeline,
    }


@app.post("/detect/video")
async def detect_video(file: UploadFile, conf: float = 0.25, classes: str | None = None,
                       max_frames: int | None = None) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video format '{ext or '(none)'}'. Use one of {sorted(_VIDEO_EXTS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    max_mb = Config.detect_video_max_upload_mb
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"video too large ({len(data) / 1e6:.1f} MB); limit is {max_mb:.0f} MB")

    frame_cap = max_frames if max_frames is not None else Config.detect_video_max_frames
    frame_cap = max(1, min(frame_cap, _VIDEO_MAX_FRAMES_CAP))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=str(_TMP_UPLOAD_DIR)) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return await run_in_threadpool(
            run_detection_video, tmp_path, conf, _parse_classes_param(classes), frame_cap)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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
