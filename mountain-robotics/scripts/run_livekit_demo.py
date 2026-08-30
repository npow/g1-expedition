#!/usr/bin/env python3
"""Launch the browser monitor and LiveKit voice agent as one stage demo."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")


def main() -> int:
    missing = [
        key for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
        if not os.environ.get(key)
    ]
    if missing:
        print("LiveKit credentials are missing: " + ", ".join(missing), file=sys.stderr)
        print("Put the hackathon project credentials in .env.local, then retry.",
              file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    monitor = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "voice_demo.py"), "--open"],
        cwd=ROOT,
        env=env,
    )
    try:
        for _ in range(50):
            if monitor.poll() is not None:
                return monitor.returncode or 1
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            print("The local control monitor did not start.", file=sys.stderr)
            return 1

        print("Starting LiveKit microphone session…")
        return subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "livekit_agent.py"), "console"],
            cwd=ROOT,
            env=env,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=3)
        except subprocess.TimeoutExpired:
            monitor.kill()


if __name__ == "__main__":
    raise SystemExit(main())
