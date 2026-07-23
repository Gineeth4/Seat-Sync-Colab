"""
Smart Seat Sync — service launcher
Run the FastAPI occupancy service from the project root, CWD-independent.

    py -3.10 serve.py                      # 127.0.0.1:8000
    py -3.10 serve.py --host 0.0.0.0 --port 8080
    py -3.10 serve.py --reload             # dev auto-reload

Env vars (see service/app.py) tune weights, source, thresholds, stream FPS.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
# Anchor CWD to the project root so relative asset writes never hit a synced dir.
os.chdir(_PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Smart Seat Sync service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("service.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    # Uniform Windows guard (uvicorn reload spawns a child process).
    raise SystemExit(main())
