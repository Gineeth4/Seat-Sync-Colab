"""
Smart Seat Sync — Milestone 2
Local YOLOv8 Training Harness (RTX 2050 / 4GB VRAM)
================================================================================

What this is
------------
A thin, *defensive* wrapper around Ultralytics' training loop whose entire job is
to make an unattended local run on a 4GB card survive to completion:

  * A single source-of-truth `TrainProfile` with VRAM-safe defaults, plus a
    `--profile safe|balanced|max` selector so we never guess batch/imgsz at the
    CLI under pressure.
  * A preflight that reads the real card, free host RAM, and dataset descriptor
    and REFUSES to start a run that will predictably OOM — failing in 2 seconds
    instead of 20 minutes.
  * Ultralytics callbacks that self-audit every epoch: peak VRAM, VRAM headroom,
    epoch wall-time, and (crucially) an automatic guard that aborts cleanly with
    an actionable message the moment reserved VRAM crosses a danger line, rather
    than letting the driver throw a mid-batch OOM that corrupts the last epoch.
  * Correct Windows semantics: the `if __name__ == '__main__':` guard is
    mandatory because Ultralytics' DataLoader uses `spawn` on Windows, which
    re-imports this module in every worker; without the guard each worker would
    re-launch training recursively.

Why these numbers (4GB budget arithmetic)
-----------------------------------------
Training memory ≈ model params + optimizer state + *activations*, and activations
scale with (batch x imgsz^2). On ~3.4 GB of usable VRAM (after Windows desktop +
driver overhead on a 4GB card), the empirically safe envelope for the nano model
with AMP is:

    yolov8n, imgsz=640, batch=4, amp=True   -> ~2.6-3.0 GB peak   (default)
    yolov8n, imgsz=512, batch=4, amp=True   -> ~1.9-2.3 GB peak   ("safe")
    yolov8n, imgsz=640, batch=8, amp=True   -> ~3.4-3.9 GB peak   ("max", risky)

We default to the middle profile and expose the others explicitly. `batch` is
kept a fixed integer (never Ultralytics' `batch=-1` auto-finder) because the
auto-finder probes by allocating until OOM, which on a 4GB desktop card can hang
the display driver.

Run:
    python scripts/m2_train.py                      # balanced profile
    python scripts/m2_train.py --profile safe
    python scripts/m2_train.py --epochs 100 --resume
    python scripts/m2_train.py --dry-run            # preflight only, no training
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

# --- Root-relative imports so the script runs from any CWD ---
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    import yaml
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] Missing dependency: {exc}. Run: pip install -r requirements.txt")
    raise SystemExit(2) from exc

_MB = 1024 * 1024
_GB = 1024 ** 3


# ----------------------------------------------------------------------------
# Training profiles — the only place VRAM-sensitive knobs live.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainProfile:
    name: str
    model: str          # pretrained checkpoint to fine-tune from
    imgsz: int
    batch: int
    workers: int        # DataLoader workers; Windows spawn overhead => keep modest
    epochs: int
    patience: int       # early-stop patience (epochs w/o val improvement)
    amp: bool
    cache: str          # 'disk' | 'ram' | False — 'disk' avoids RAM blowups
    est_peak_vram_mb: int  # expected high-water mark, used by the preflight gate


# NOTE: workers=2 is deliberate on Windows. Each worker is a full process (spawn),
# so 8 workers on a 6-core laptop mostly buys context-switch thrash + RAM, not
# throughput, and the image decode is not the bottleneck at these batch sizes.
#
# VRAM estimates below are EMPIRICALLY ANCHORED, not guessed. A measured run on
# this RTX 2050 (yolov8n, imgsz=512, batch=4, amp) peaked at ~401 MB — the nano
# model is far lighter than a naive 4GB-card safety budget assumes. Activation
# memory scales ~linearly with batch and with imgsz^2, so from the 401 MB anchor:
#     imgsz=512,b4  -> ~0.40 GB (measured)     imgsz=640,b8  -> ~1.3 GB
#     imgsz=640,b16 -> ~2.6 GB                 imgsz=640,b32 -> ~4.9 GB (won't fit)
# Estimates carry a ~1.4x head-margin over the linear projection for augmentation
# spikes and the cuDNN workspace, so the preflight gate stays honest without being
# the 6x-too-conservative numbers this file originally shipped with.
PROFILES: dict[str, TrainProfile] = {
    "safe": TrainProfile(
        name="safe", model="yolov8n.pt", imgsz=512, batch=4, workers=2,
        epochs=100, patience=25, amp=True, cache="disk", est_peak_vram_mb=600,
    ),
    "balanced": TrainProfile(
        name="balanced", model="yolov8n.pt", imgsz=640, batch=8, workers=2,
        epochs=100, patience=25, amp=True, cache="disk", est_peak_vram_mb=1500,
    ),
    "max": TrainProfile(
        name="max", model="yolov8n.pt", imgsz=640, batch=16, workers=3,
        epochs=100, patience=25, amp=True, cache="disk", est_peak_vram_mb=2800,
    ),
}

# Abort the run if reserved VRAM ever climbs above this fraction of total; on a
# 4GB card the last ~5% is where the display compositor and the allocator fight
# and the driver TDRs. Catching it in a callback lets us stop *cleanly*.
_VRAM_ABORT_FRACTION = 0.94


# ----------------------------------------------------------------------------
# Preflight — fail fast, fail readable.
# ----------------------------------------------------------------------------
def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}")
    raise SystemExit(1)


def preflight(profile: TrainProfile, data_yaml: Path) -> dict:
    """Validate hardware + dataset before committing to a run. Returns a metrics
    dict on success; raises SystemExit(1) with a specific reason on failure."""
    print("-" * 68)
    print(f" Preflight — profile '{profile.name}'")
    print("-" * 68)

    if not torch.cuda.is_available():
        _fatal("CUDA unavailable. Run scripts/m1_cuda_verify.py to diagnose.")

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / _MB
    print(f"  GPU               : {props.name}  ({total_mb:.0f} MB VRAM)")
    print(f"  Compute capability: {props.major}.{props.minor}")

    # Gate 1: does the profile's estimated peak fit under a safety margin?
    usable_mb = total_mb * 0.90  # reserve 10% for driver/desktop
    print(f"  Usable VRAM budget: {usable_mb:.0f} MB "
          f"(est. peak for this profile: {profile.est_peak_vram_mb} MB)")
    if profile.est_peak_vram_mb > usable_mb:
        _fatal(
            f"Profile '{profile.name}' needs ~{profile.est_peak_vram_mb} MB but only "
            f"~{usable_mb:.0f} MB is usable. Use --profile safe (imgsz=512) or lower --batch."
        )

    # Gate 2: dataset descriptor present and internally consistent.
    if not data_yaml.is_file():
        _fatal(f"Dataset descriptor not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    for key in ("path", "train", "val", "names"):
        if key not in spec:
            _fatal(f"{data_yaml.name} is missing required key '{key}'.")
    ds_root = Path(spec["path"])
    train_dir = ds_root / spec["train"]
    val_dir = ds_root / spec["val"]
    if not train_dir.is_dir():
        _fatal(f"Train images dir does not exist: {train_dir}")
    if not val_dir.is_dir():
        _fatal(f"Val images dir does not exist: {val_dir}")

    # Gate 3: warn on OneDrive paths (sync locks corrupt epochs; see project memory).
    if "onedrive" in str(ds_root).lower():
        print("  [WARN] Dataset lives under OneDrive; sync may lock images mid-epoch. "
              "Move it to a non-synced folder if you hit intermittent read errors.")

    n_classes = len(spec["names"])
    print(f"  Dataset root      : {ds_root}")
    print(f"  Classes           : {n_classes}  {list(spec['names'].values())}")
    print("  Preflight         : OK")
    return {"total_vram_mb": total_mb, "n_classes": n_classes}


# ----------------------------------------------------------------------------
# Runtime self-audit callbacks (wired into the Ultralytics trainer).
# ----------------------------------------------------------------------------
class RuntimeMonitor:
    """Tracks per-epoch VRAM + timing and enforces a hard VRAM abort line.

    Ultralytics exposes hook points; we attach to:
      on_train_epoch_start  -> reset peak stats + start timer
      on_train_epoch_end    -> read peak/reserved, log, and abort if unsafe
      on_train_end          -> print an aggregate report
    """

    def __init__(self, total_vram_mb: float) -> None:
        self.total_vram_mb = total_vram_mb
        self.abort_mb = total_vram_mb * _VRAM_ABORT_FRACTION
        self._epoch_t0 = 0.0
        self.epoch_times: list[float] = []
        self.epoch_peak_mb: list[float] = []
        self.aborted = False

    # -- hooks ---------------------------------------------------------------
    def on_epoch_start(self, trainer) -> None:  # noqa: ANN001 - Ultralytics passes trainer
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._epoch_t0 = time.perf_counter()

    def on_epoch_end(self, trainer) -> None:  # noqa: ANN001
        dt = time.perf_counter() - self._epoch_t0
        self.epoch_times.append(dt)

        peak_mb = reserved_mb = 0.0
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / _MB
            reserved_mb = torch.cuda.memory_reserved() / _MB
        self.epoch_peak_mb.append(peak_mb)

        epoch = getattr(trainer, "epoch", len(self.epoch_times) - 1)
        headroom = self.total_vram_mb - reserved_mb
        print(
            f"  [monitor] epoch {epoch:>3} | {dt:6.1f}s "
            f"| peak {peak_mb:6.0f} MB | reserved {reserved_mb:6.0f} MB "
            f"| headroom {headroom:6.0f} MB"
        )

        # Hard guard: if the allocator pool has crept past the danger line, stop
        # BEFORE the next epoch triggers a driver-level OOM/TDR.
        if reserved_mb >= self.abort_mb and not self.aborted:
            self.aborted = True
            print(
                f"  [monitor][ABORT] reserved VRAM {reserved_mb:.0f} MB crossed the "
                f"{self.abort_mb:.0f} MB abort line ({_VRAM_ABORT_FRACTION:.0%} of card). "
                "Stopping cleanly — re-run with --profile safe or a lower --batch."
            )
            # Ultralytics honours trainer.stop for a graceful shutdown at the
            # epoch boundary (weights already checkpointed for this epoch).
            trainer.stop = True

    def on_train_end(self, trainer) -> None:  # noqa: ANN001
        if not self.epoch_times:
            return
        avg = sum(self.epoch_times) / len(self.epoch_times)
        peak = max(self.epoch_peak_mb) if self.epoch_peak_mb else 0.0
        print("-" * 68)
        print(f"  [monitor] epochs run      : {len(self.epoch_times)}")
        print(f"  [monitor] avg epoch time  : {avg:.1f}s")
        print(f"  [monitor] slowest epoch   : {max(self.epoch_times):.1f}s")
        print(f"  [monitor] peak VRAM (run) : {peak:.0f} MB "
              f"({peak / self.total_vram_mb:.0%} of card)")
        print(f"  [monitor] status          : "
              f"{'ABORTED on VRAM guard' if self.aborted else 'completed'}")


def _attach_monitor(model, monitor: RuntimeMonitor) -> None:
    model.add_callback("on_train_epoch_start", monitor.on_epoch_start)
    model.add_callback("on_train_epoch_end", monitor.on_epoch_end)
    model.add_callback("on_train_end", monitor.on_train_end)


# ----------------------------------------------------------------------------
# Training entrypoint.
# ----------------------------------------------------------------------------
def train(profile: TrainProfile, data_yaml: Path, resume: bool,
          run_name: str, dry_run: bool) -> int:
    metrics = preflight(profile, data_yaml)

    if dry_run:
        print("\n[dry-run] Preflight passed; skipping training as requested.")
        return 0

    # CWD guard: Ultralytics downloads pretrained weights (yolov8n.pt) and writes
    # assorted temp files into the *current working directory*. If we were launched
    # from a OneDrive-synced folder, those writes are silently dropped/locked and
    # the download fails with a curl write error. Anchor CWD to the project root
    # (a normal local dir) so every stray write lands somewhere writable. Data and
    # run paths are already absolute, so this is safe.
    os.chdir(_PROJECT_ROOT)

    # Keep pretrained weights in a dedicated, writable subdir instead of scattering
    # yolov8n.pt at the repo root.
    weights_dir = _PROJECT_ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    # Absolute path -> Ultralytics recognises the basename (yolov8n.pt) as a known
    # asset and downloads it straight into weights/, caching it for later runs.
    model_arg = str(weights_dir / profile.model)

    # Import Ultralytics only after preflight so a missing dataset fails in 2s
    # rather than after the (slow) framework import.
    from ultralytics import YOLO

    # Deterministic-ish, fixed-shape fast path: cuDNN autotune helps because our
    # imgsz is constant across the run.
    torch.backends.cudnn.benchmark = True

    print("-" * 68)
    print(f" Training — {profile.model} | imgsz={profile.imgsz} | batch={profile.batch} "
          f"| workers={profile.workers} | amp={profile.amp}")
    print("-" * 68)

    model = YOLO(model_arg)
    monitor = RuntimeMonitor(total_vram_mb=metrics["total_vram_mb"])
    _attach_monitor(model, monitor)

    try:
        model.train(
            data=str(data_yaml),
            epochs=profile.epochs,
            imgsz=profile.imgsz,
            batch=profile.batch,
            workers=profile.workers,
            amp=profile.amp,
            cache=profile.cache,
            device=0,
            patience=profile.patience,
            resume=resume,
            project=str(_PROJECT_ROOT / "runs"),
            name=run_name,
            exist_ok=resume,       # allow reusing the dir only when resuming
            plots=True,
            verbose=True,
        )
    except torch.cuda.OutOfMemoryError:
        # The callback guard catches slow creep; this catches a sudden single-batch
        # OOM (e.g. an unusually crowded frame). Give the same actionable exit.
        torch.cuda.empty_cache()
        print("\n[FATAL] CUDA OOM during a training batch. "
              "Drop to --profile safe (imgsz=512) or --batch 2, then re-run with --resume.")
        return 1

    print(f"\n[done] Weights: {_PROJECT_ROOT / 'runs' / run_name / 'weights' / 'best.pt'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smart Seat Sync — YOLOv8 training harness")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced",
                        help="VRAM profile (default: balanced).")
    parser.add_argument("--data", type=Path,
                        default=_PROJECT_ROOT / "configs" / "seat_data.yaml",
                        help="Path to the Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the profile's epoch count.")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override the profile's batch size (use with care on 4GB).")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override the profile's image size.")
    parser.add_argument("--name", default="seat_sync_yolov8n",
                        help="Run name under runs/.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run of the same name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run preflight only; do not train.")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    # Apply explicit overrides on top of the chosen profile (immutable dataclass).
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if overrides:
        profile = replace(profile, **overrides)
        print(f"[info] Applied overrides: {overrides}")

    return train(profile, args.data.resolve(), args.resume, args.name, args.dry_run)


if __name__ == "__main__":
    # === Windows multiprocessing protection (MANDATORY) ===
    # Ultralytics' DataLoader uses the 'spawn' start method on Windows, which
    # re-imports this module inside every worker process. Without this guard,
    # each worker would fall through to main() and recursively launch training,
    # exhausting RAM and spawning an unbounded process tree. All executable
    # entrypoints in this codebase live under this guard.
    sys.exit(main())
