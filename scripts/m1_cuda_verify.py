"""
Smart Seat Sync — Milestone 1
Local Environment Verification & RTX 2050 PyTorch/CUDA Test Pipeline
====================================================================

Purpose
-------
Deterministically prove that this specific machine (Windows 11 / RTX 2050 / 4GB
VRAM) can run the full training + inference stack BEFORE we spend an hour on a
YOLO26 training run that dies at epoch 3 with a CUDA OOM.

Design intent
-------------
Every check is a small, isolated, *self-auditing* stage. Each stage:
  * asserts a hard precondition (raises on fatal, warns on degraded),
  * measures the resource it exercises (VRAM allocated / reserved, wall time),
  * releases its allocations before the next stage so we measure headroom, not
    accumulated leakage.

The final stage simulates a real inference workload (fixed-size FP16 forward
passes) and reports achievable FPS + a per-frame VRAM ceiling, which is the
number we actually budget the pipeline against.

Run:
    python scripts/m1_cuda_verify.py
    python scripts/m1_cuda_verify.py --strict   # non-zero exit on any WARN
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

# --- Third-party imports are wrapped so a missing/CPU-only install produces a
#     human-readable diagnosis instead of a raw ImportError traceback. ---
try:
    import torch
except ImportError as exc:  # pragma: no cover - environment guard
    print("[FATAL] PyTorch is not importable. Install the CUDA build first:")
    print("        pip install torch==2.3.1 torchvision==0.18.1 "
          "--index-url https://download.pytorch.org/whl/cu121")
    raise SystemExit(2) from exc

try:
    import psutil
except ImportError:  # psutil is optional; degrade gracefully.
    psutil = None


# ----------------------------------------------------------------------------
# Result plumbing
# ----------------------------------------------------------------------------
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

_STATUS_GLYPH = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    metrics: dict = field(default_factory=dict)


class Auditor:
    """Collects stage results and renders a single machine-readable report."""

    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.results: list[CheckResult] = []

    def record(self, result: CheckResult) -> None:
        self.results.append(result)
        line = f"{_STATUS_GLYPH[result.status]} {result.name}"
        if result.detail:
            line += f" — {result.detail}"
        print(line)
        for k, v in result.metrics.items():
            print(f"         {k}: {v}")

    def run_stage(self, name: str, fn: Callable[[], CheckResult]) -> None:
        """Execute a stage, converting any escaped exception into a FAIL row.

        This is the outermost self-audit: a stage that raises unexpectedly must
        never abort the whole report, or we lose the diagnostics from the stages
        that already passed.
        """
        try:
            self.record(fn())
        except torch.cuda.OutOfMemoryError as exc:
            self.record(CheckResult(name, FAIL, f"CUDA OOM: {exc}"))
        except AssertionError as exc:
            self.record(CheckResult(name, FAIL, str(exc) or "assertion failed"))
        except Exception as exc:  # noqa: BLE001 - report-and-continue is intentional
            self.record(CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}"))
        finally:
            _free_cuda()

    def exit_code(self) -> int:
        if any(r.status == FAIL for r in self.results):
            return 1
        if self.strict and any(r.status == WARN for r in self.results):
            return 1
        return 0


# ----------------------------------------------------------------------------
# CUDA memory helpers
# ----------------------------------------------------------------------------
_MB = 1024 * 1024


def _free_cuda() -> None:
    """Return VRAM to the driver so the next stage measures true headroom."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _vram_snapshot() -> dict:
    """Current allocation picture in MB (allocated = live tensors,
    reserved = caching-allocator pool, peak = high-water mark since last reset)."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / _MB, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / _MB, 1),
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / _MB, 1),
    }


# ----------------------------------------------------------------------------
# Stage 1 — Host / interpreter fingerprint
# ----------------------------------------------------------------------------
def check_host() -> CheckResult:
    metrics = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        metrics["host_ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
        metrics["host_ram_avail_gb"] = round(vm.available / (1024 ** 3), 1)
        if vm.available < 4 * 1024 ** 3:
            return CheckResult(
                "Host fingerprint", WARN,
                "Less than 4 GB host RAM free; close apps before training.",
                metrics,
            )
    return CheckResult("Host fingerprint", PASS, metrics=metrics)


# ----------------------------------------------------------------------------
# Stage 2 — CUDA / cuDNN availability
# ----------------------------------------------------------------------------
def check_cuda_available() -> CheckResult:
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False. You almost certainly installed the "
        "CPU-only wheel. Reinstall torch from the cu121 index (see requirements.txt)."
    )
    metrics = {
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "device_count": torch.cuda.device_count(),
    }
    assert torch.backends.cudnn.is_available(), "cuDNN is not available to torch."
    return CheckResult("CUDA & cuDNN available", PASS, metrics=metrics)


# ----------------------------------------------------------------------------
# Stage 3 — GPU identity + VRAM budget sanity
# ----------------------------------------------------------------------------
def check_device_properties() -> CheckResult:
    props = torch.cuda.get_device_properties(0)
    total_mb = round(props.total_memory / _MB, 1)
    metrics = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": total_mb,
        "multiprocessors": props.multi_processor_count,
    }
    # RTX 2050 reports ~4096 MB. Anything under ~3.4 GB usable means we must drop
    # to imgsz=512 / batch=2 in Milestone 2, so surface it as a WARN not a PASS.
    if total_mb < 3400:
        return CheckResult(
            "GPU properties", WARN,
            f"Only {total_mb} MB VRAM detected; use the low-VRAM training profile.",
            metrics,
        )
    return CheckResult("GPU properties", PASS, metrics=metrics)


# ----------------------------------------------------------------------------
# Stage 4 — Compute correctness (not just "does it run", but "is it right")
# ----------------------------------------------------------------------------
def check_matmul_correctness() -> CheckResult:
    """A GPU that runs but returns garbage (bad driver/thermal) is worse than one
    that fails loudly. Verify numerical agreement against a CPU reference."""
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    cpu_ref = a @ b
    gpu_out = (a.cuda() @ b.cuda()).cpu()
    max_abs_err = (cpu_ref - gpu_out).abs().max().item()
    assert torch.allclose(cpu_ref, gpu_out, atol=1e-3), (
        f"GPU matmul disagrees with CPU (max abs err {max_abs_err:.2e}). "
        "Suspect an unstable overclock, thermal throttling, or a driver mismatch."
    )
    return CheckResult(
        "Compute correctness", PASS,
        metrics={"max_abs_err_vs_cpu": f"{max_abs_err:.2e}"},
    )


# ----------------------------------------------------------------------------
# Stage 5 — AMP forward/backward (the exact code path training uses)
# ----------------------------------------------------------------------------
def check_amp_backward() -> CheckResult:
    """Milestone 2 trains with mixed precision (amp=True). Prove autocast +
    GradScaler + backward works and measure the VRAM one small step costs."""
    device = torch.device("cuda")
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 3),
    ).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    # torch 2.x API: device-typed GradScaler (the old torch.cuda.amp.GradScaler
    # is deprecated). Falls back gracefully on older torch that lacks torch.amp.
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(2, 3, 640, 640, device=device)
    y = torch.randint(0, 3, (2,), device=device)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(net(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()

    assert torch.isfinite(loss).item(), "Loss is NaN/Inf under AMP — bad autocast path."
    return CheckResult(
        "AMP forward/backward", PASS,
        detail=f"loss={loss.item():.4f}",
        metrics=_vram_snapshot(),
    )


# ----------------------------------------------------------------------------
# Stage 6 — VRAM ceiling probe (how much can we actually allocate?)
# ----------------------------------------------------------------------------
def check_vram_ceiling() -> CheckResult:
    """Allocate in 128 MB blocks until the allocator refuses, then report the
    largest contiguous headroom. This is the empirical batch-size budget."""
    blocks: list[torch.Tensor] = []
    block_elems = 128 * _MB // 4  # 128 MB of float32
    allocated_mb = 0.0
    try:
        while True:
            blocks.append(torch.empty(block_elems, dtype=torch.float32, device="cuda"))
            allocated_mb += 128
            if allocated_mb >= 4096:  # never probe past the card's nominal size
                break
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        del blocks  # drop refs before empty_cache in the stage runner

    # We expect to reach at least ~2.4 GB of user allocations on a 4GB card after
    # driver/display overhead; less than that means the desktop is eating VRAM.
    status = PASS if allocated_mb >= 2400 else WARN
    detail = (
        f"Reached {allocated_mb:.0f} MB of user allocations before OOM/limit."
        if status == PASS else
        f"Only {allocated_mb:.0f} MB allocatable — reduce display load / other GPU apps."
    )
    return CheckResult("VRAM ceiling probe", status, detail,
                       metrics={"max_user_alloc_mb": allocated_mb})


# ----------------------------------------------------------------------------
# Stage 7 — Inference throughput + per-frame VRAM (the pipeline's real budget)
# ----------------------------------------------------------------------------
def check_inference_throughput() -> CheckResult:
    """Simulate the deployed detector: fixed 640x640 FP16 forward passes through
    a small conv stack, batch=1, no grad. Report FPS and the per-frame VRAM
    high-water mark. Includes a warmup so we don't time cuDNN autotuning."""
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # let cuDNN pick fast kernels for a fixed shape

    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, stride=2, padding=1), torch.nn.SiLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.SiLU(),
        torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.SiLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(64, 3),
    ).to(device).eval()

    frame = torch.randn(1, 3, 640, 640, device=device)
    n_warmup, n_iters = 5, 60
    dropped = 0
    latencies_ms: list[float] = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            net(frame)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            t0 = time.perf_counter()
            net(frame)
            torch.cuda.synchronize()  # forward is async; sync to time it honestly
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            # Self-audit: a frame slower than a 15 FPS budget (66.7 ms) is "dropped".
            if dt_ms > 66.7:
                dropped += 1

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
    fps = 1000.0 / (sum(latencies_ms) / len(latencies_ms))

    metrics = {
        "fps_mean": round(fps, 1),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "frames_over_66ms": f"{dropped}/{n_iters}",
        **_vram_snapshot(),
    }
    status = PASS if dropped == 0 else WARN
    detail = ("Sustained >15 FPS on the synthetic detector."
              if status == PASS else
              f"{dropped} frame(s) exceeded the 66.7 ms budget under synthetic load.")
    return CheckResult("Inference throughput", status, detail, metrics)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RTX 2050 CUDA verification pipeline")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any stage returns WARN.")
    args = parser.parse_args(argv)

    print("=" * 68)
    print(" Smart Seat Sync — Milestone 1: Local CUDA Verification")
    print("=" * 68)

    auditor = Auditor(strict=args.strict)

    # Ordered: cheap host checks first, then CUDA gating, then the heavy probes.
    auditor.run_stage("Host fingerprint", check_host)
    auditor.run_stage("CUDA & cuDNN available", check_cuda_available)

    # If CUDA itself is unavailable, the GPU stages cannot run — short-circuit
    # with an explicit FAIL row per stage so the report stays complete.
    cuda_ok = auditor.results[-1].status == PASS
    gpu_stages = [
        ("GPU properties", check_device_properties),
        ("Compute correctness", check_matmul_correctness),
        ("AMP forward/backward", check_amp_backward),
        ("VRAM ceiling probe", check_vram_ceiling),
        ("Inference throughput", check_inference_throughput),
    ]
    for name, fn in gpu_stages:
        if cuda_ok:
            auditor.run_stage(name, fn)
        else:
            auditor.record(CheckResult(name, FAIL, "skipped: CUDA unavailable"))

    print("-" * 68)
    n_pass = sum(r.status == PASS for r in auditor.results)
    n_warn = sum(r.status == WARN for r in auditor.results)
    n_fail = sum(r.status == FAIL for r in auditor.results)
    print(f" Summary: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")
    code = auditor.exit_code()
    print(f" Exit code: {code} "
          f"({'GO for Milestone 2' if code == 0 else 'resolve failures before training'})")
    print("=" * 68)
    return code


if __name__ == "__main__":
    # Windows guard: this script spawns no workers, but we keep the idiom uniform
    # across the codebase so nothing here is copy-pasted into a DataLoader context
    # without the protection already present.
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
