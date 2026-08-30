#!/usr/bin/env python3
"""Local voice-command console for the coordinated log lift.

The browser supplies speech recognition and this process owns the command
boundary, MuJoCo mission, live renderer, and telemetry.  It deliberately uses
the lift snapshot in ``_hfjob``: the top-level package is a newer roll/push
experiment while the README, recorded submission, and requested demo are the
coordinated lift.

    PYTHONPATH=. .venv/bin/python scripts/voice_demo.py --open

Say "lift the log", use the on-screen fallback button, or POST the same phrase
to /api/command.  A LiveKit agent can use that endpoint as a function tool.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LIFT_SNAPSHOT = ROOT / "_hfjob"
DEMO_PAGE = ROOT / "demo" / "voice.html"
POSTER = ROOT / "out" / "frame_HOLD.png"
RECORDED_DEMO = ROOT / "out" / "lifting_log.mp4"

# The lift implementation was captured with the training job.  Put it before
# the newer top-level experiment without overwriting either body of work.
sys.path.insert(0, str(LIFT_SNAPSHOT))
os.environ.setdefault("MUJOCO_GL", "glfw")


LIFT_VERBS = {"lift", "raise", "hoist", "pickup", "pick up"}
LOG_NOUNS = {"log", "trunk", "timber", "tree", "it", "obstacle"}
STOP_WORDS = {"stop", "abort", "halt", "stand down", "emergency stop"}
GUST_VERBS = {"add", "apply", "simulate", "inject", "trigger", "start"}
GUST_NOUNS = {"wind", "gust", "blizzard", "gale"}
ICE_WORDS = {"verglas", "ice", "icy", "black ice", "frost", "frozen", "slippery"}
HEAVY_WORDS = {"heavy", "overweight", "thirty kilo", "30kg", "30 kg", "too heavy", "decline"}
STATUS_WORDS = {"status", "telemetry", "report", "readout", "diagnostics", "read out"}
RESET_WORDS = {"reset", "restart", "re arm", "rearm", "ready stance", "re-arm"}


def _normalise(phrase: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", phrase.lower()).strip()


def classify_intent(phrase: str) -> str | None:
    """Return the one safe command represented by a transcript, if any."""
    clean = " ".join(_normalise(phrase).split())
    if not clean:
        return None
    padded = f" {clean} "
    contains = lambda words: any(f" {word} " in padded for word in words)

    # 1. Stop always has top priority for safety
    if contains(STOP_WORDS):
        return "stop"

    # 2. Reset / re-arm system
    if contains(RESET_WORDS) or clean in ("ready stance", "rearm system", "system reset"):
        return "reset"

    # 3. Status queries
    if (
        contains(STATUS_WORDS)
        or any(clean.startswith(prefix) for prefix in ("what is the status", "what is your status", "check status", "tell me status", "how are you", "system status"))
    ):
        return "status"

    # 4. Heavy load test (Track 3 Thinking: Autonomous weigh-in and safe no-go)
    if (
        contains(HEAVY_WORDS)
        and (contains(LIFT_VERBS) or any(w in clean for w in ("log", "trunk", "timber", "load", "test", "evaluate", "weigh", "obstacle")))
    ) or clean in ("test heavy load", "simulate heavy log", "decline lift", "heavy load test"):
        return "heavy_load"

    # 5. Verglas ice disturbance (Track 1 Movement: Walking on ice friction drop)
    if (
        contains(ICE_WORDS)
        and (contains(GUST_VERBS) or any(w in clean for w in ("underfoot", "terrain", "condition", "test", "surface", "rock", "ground", "slip", "sliding")))
    ) or any(clean.startswith(prefix) for prefix in ("simulate verglas", "simulate ice", "apply ice", "verglas underfoot", "black ice", "icy terrain", "add ice", "ice test")):
        return "verglas_ice"

    # 6. Safety check: "Don't lift the log" is not a lift command
    if any(
        f" {negative} {verb} " in padded
        for negative in ("do not", "don t", "never", "not")
        for verb in LIFT_VERBS
    ):
        return None

    # 7. Wind disturbance (45 N lateral disturbance)
    if contains(GUST_VERBS) and contains(GUST_NOUNS):
        return "wind_gust"

    # 8. Nominal coordinated lift
    has_verb = contains(LIFT_VERBS)
    has_object = any(re.search(rf"\b{re.escape(noun)}\b", clean) for noun in LOG_NOUNS)
    return "lift_log" if has_verb and has_object else None


@dataclass
class DemoState:
    speed: float = 1.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None
    stop_requested: bool = False
    gust_requested: bool = False
    ice_requested: bool = False
    active_payload_mass: float = 12.0
    frame: bytes | None = None
    frame_seq: int = 0
    data: dict[str, Any] = field(default_factory=lambda: {
        "state": "armed",
        "mode": "live physics",
        "message": "Say “lift the log”",
        "heard": "",
        "phase": "READY",
        "elapsed": 0.0,
        "lift_cm": 0.0,
        "tilt_deg": 0.0,
        "share_a": 50.0,
        "mass_kg": 0.0,
        "peak_force_n": 0.0,
        "go": True,
        "go_reason": "waiting for operator command",
        "events": [],
        "frame_seq": 0,
        "gust_status": "standby",
        "wind_n": 0.0,
        "terrain_status": "firm rock (μ=0.9)",
        "livekit_connected": False,
        "voice_source": "browser fallback",
    })

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.data)

    def update(self, **values: Any) -> None:
        with self.lock:
            self.data.update(values)

    def prepare_preview(self) -> None:
        """Render the real READY pose so the armed console never shows a fake run."""
        recorder = None
        try:
            from PIL import Image

            from alpine_lift.mission import Mission
            from alpine_lift.render import Recorder

            mission = Mission()
            recorder = Recorder(
                mission.model,
                width=960,
                height=540,
                shot="wide",
                title="VOICE-COMMANDED ALPINE LIFT",
            )
            tele = mission.step()
            pixels = recorder.frame(
                mission.data,
                tele,
                {
                    "go_reason": "waiting for operator command",
                    "sling_ok": mission.ctrl.coord.sling_ok,
                    "sling_limit": mission.scfg.sling_strength,
                },
            )
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="JPEG", quality=82)
            with self.lock:
                if self.data["state"] == "armed":
                    self.frame = buf.getvalue()
                    self.frame_seq += 1
                    self.data["frame_seq"] = self.frame_seq
        except Exception:
            # The bundled poster remains a deterministic zero-setup fallback.
            return
        finally:
            if recorder is not None:
                recorder.close()

    def reset_state(self, phrase: str = "reset") -> tuple[bool, str]:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                self.stop_requested = True
            self.gust_requested = False
            self.ice_requested = False
            self.active_payload_mass = 12.0
            self.data.update({
                "state": "armed",
                "message": "System re-armed. Say “lift the log” or “test heavy load”.",
                "heard": phrase,
                "phase": "READY",
                "elapsed": 0.0,
                "lift_cm": 0.0,
                "tilt_deg": 0.0,
                "share_a": 50.0,
                "mass_kg": 0.0,
                "peak_force_n": 0.0,
                "go": True,
                "go_reason": "waiting for operator command",
                "events": [],
                "gust_status": "standby",
                "wind_n": 0.0,
                "terrain_status": "firm rock (μ=0.9)",
            })
        threading.Thread(target=self.prepare_preview, daemon=True).start()
        return True, "System reset to armed ready stance."

    def start_lift(self, phrase: str, payload_mass: float = 12.0) -> tuple[bool, str]:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                self.data["heard"] = phrase
                return True, "Lift is already in progress."
            self.stop_requested = False
            self.gust_requested = False
            self.ice_requested = False
            self.active_payload_mass = payload_mass
            self.frame = None
            self.frame_seq = 0
            label = "heavy 30 kg log" if payload_mass > 20.0 else "standard log"
            self.data.update({
                "state": "starting",
                "mode": "live physics",
                "message": f"Command confirmed — evaluating {label}",
                "heard": phrase,
                "phase": "READY",
                "elapsed": 0.0,
                "lift_cm": 0.0,
                "tilt_deg": 0.0,
                "share_a": 50.0,
                "mass_kg": 0.0,
                "peak_force_n": 0.0,
                "go": True,
                "go_reason": "not yet weighed",
                "events": [],
                "frame_seq": 0,
                "gust_status": "standby",
                "wind_n": 0.0,
                "terrain_status": "firm rock (μ=0.9)",
            })
            self.worker = threading.Thread(
                target=self._run_lift,
                args=(payload_mass,),
                name="alpine-lift-demo",
                daemon=True,
            )
            self.worker.start()
        if payload_mass > 20.0:
            return True, "Evaluating 30 kilogram heavy obstacle. Weighing in progress."
        return True, "Lift command accepted."

    def request_stop(self, phrase: str) -> tuple[bool, str]:
        with self.lock:
            running = self.worker is not None and self.worker.is_alive()
            self.data["heard"] = phrase
            if running:
                self.stop_requested = True
                self.data["message"] = "Operator stop received — controlled set-down"
                return True, "Controlled set-down requested."
        return False, "No lift is currently running."

    def request_gust(self, phrase: str) -> tuple[bool, str]:
        with self.lock:
            running = self.worker is not None and self.worker.is_alive()
            self.data["heard"] = phrase
            if running:
                self.gust_requested = True
                self.data.update({
                    "gust_status": "armed",
                    "message": "Voice gust armed — waiting for a loaded phase",
                })
                return True, "A 45 newton wind gust is armed for the loaded phase."
        return False, "Start the lift before adding a wind gust."

    def request_ice(self, phrase: str) -> tuple[bool, str]:
        with self.lock:
            running = self.worker is not None and self.worker.is_alive()
            self.data["heard"] = phrase
            if running:
                self.ice_requested = True
                self.data.update({
                    "terrain_status": "verglas ice armed (μ=0.45)",
                    "message": "Verglas warning armed — terrain friction will drop to 0.45",
                })
                return True, "Verglas surface ice disturbance armed for the lift."
        return False, "Start the lift before applying surface verglas ice."

    def report_telemetry(self, phrase: str) -> tuple[bool, str]:
        with self.lock:
            snap = dict(self.data)
            snap["heard"] = phrase
            self.data["heard"] = phrase
            phase = snap.get("phase", "READY")
            mass = snap.get("mass_kg", 0.0)
            tilt = snap.get("tilt_deg", 0.0)
            share_a = round(snap.get("share_a", 50.0))
            force = round(snap.get("peak_force_n", 0.0))
            status_summary = (
                f"Phase is {phase}. Measured mass is {mass:.1f} kg. Tilt is {tilt:.1f} degrees. "
                f"Load share is {share_a} percent robot A, {100-share_a} percent robot B. Peak force {force} newtons."
            )
            self.data["message"] = f"Telemetry: {phase} | {mass:.1f}kg | {tilt:.1f}° | {force}N"
            return True, status_summary

    def _run_lift(self, payload_mass: float = 12.0) -> None:
        recorder = None
        try:
            from PIL import Image

            from alpine_lift.mission import Mission
            from alpine_lift.render import Recorder
            from alpine_lift.scene import SceneConfig

            scene_cfg = SceneConfig(payload_mass=payload_mass)
            mission = Mission(scene=scene_cfg)
            try:
                recorder = Recorder(
                    mission.model,
                    width=960,
                    height=540,
                    shot="wide",
                    title="VOICE-COMMANDED ALPINE LIFT",
                )
                render_mode = "live physics"
            except Exception as exc:  # the physics demo still runs without GL
                render_mode = "recorded visual + live physics"
                self.update(
                    mode=render_mode,
                    message=f"Renderer unavailable ({type(exc).__name__}); using visual fallback",
                )

            self.update(state="running", mode=render_mode,
                        message="Assessing the unknown load before committing")
            tick = 0
            gust_start = None
            ice_applied = False
            next_tick = time.monotonic()
            for tele in mission.run():
                with self.lock:
                    stop = self.stop_requested
                    gust_requested = self.gust_requested
                    ice_requested = self.ice_requested
                if stop and not mission.ctrl.aborted:
                    mission.ctrl._abort("operator stop", mission.data)

                # Wind gust disturbance logic
                if (
                    gust_requested
                    and gust_start is None
                    and tele.phase in mission.dist.gust_phases
                ):
                    gust_start = tele.t
                    mission.dist.wind_gust = 45.0
                    mission.ctrl.events.append(
                        (tele.t, "voice command: 45N wind gust")
                    )
                    self.update(gust_status="45 N active", wind_n=45.0)
                elif gust_start is not None and tele.t - gust_start >= 1.8:
                    mission.dist.wind_gust = 0.0
                    gust_start = None
                    with self.lock:
                        self.gust_requested = False
                    mission.ctrl.events.append((tele.t, "wind gust cleared"))
                    self.update(gust_status="complete", wind_n=0.0)

                # Verglas surface ice disturbance logic (Track 1: Movement / Extreme terrain)
                if (
                    ice_requested
                    and not ice_applied
                    and tele.phase in ("HOLD", "SHIFT", "LIFT")
                ):
                    f = mission.model.pair_friction
                    ground_pairs = [
                        i for i in range(mission.model.npair)
                        if mission.ix.ground_geom in (mission.model.pair_geom1[i], mission.model.pair_geom2[i])
                    ]
                    for i in ground_pairs:
                        f[i, 0] = f[i, 1] = 0.45
                    ice_applied = True
                    mission.ctrl.events.append((tele.t, "voice command: verglas ice (μ=0.45)"))
                    self.update(terrain_status="verglas ice active (μ=0.45)")

                if recorder is not None and tick % 2 == 0:
                    extra = {
                        "go_reason": mission.ctrl.coord.go_reason,
                        "sling_ok": mission.ctrl.coord.sling_ok,
                        "sling_limit": mission.scfg.sling_strength,
                    }
                    pixels = recorder.frame(mission.data, tele, extra)
                    buf = io.BytesIO()
                    Image.fromarray(pixels).save(buf, format="JPEG", quality=82)
                    with self.lock:
                        self.frame = buf.getvalue()
                        self.frame_seq += 1

                go_reason = mission.ctrl.coord.go_reason
                if tele.phase in ("HOLD", "SHIFT"):
                    message = "Load accepted — coordinated lift in progress"
                elif tele.phase in ("LOWER", "RELEASE", "RETREAT"):
                    message = "Trail cleared — controlled set-down"
                elif tele.aborted:
                    message = f"Safety abort — {go_reason}"
                else:
                    message = "Measuring sling load and balance margins"
                self.update(
                    state="running",
                    message=message,
                    phase=tele.phase,
                    elapsed=round(float(tele.t), 2),
                    lift_cm=round(float(tele.payload_lift) * 100.0, 1),
                    tilt_deg=round(float(tele.tilt_deg), 2),
                    share_a=round(float(tele.share) * 100.0, 1),
                    mass_kg=round(float(tele.mass_est), 1),
                    peak_force_n=round(max(tele.hand_force.values(), default=0.0), 1),
                    go=bool(tele.go),
                    go_reason=go_reason,
                    events=[[round(float(t), 2), text] for t, text in tele.events[-4:]],
                    frame_seq=self.frame_seq,
                )

                tick += 1
                next_tick += 0.02 / max(self.speed, 1e-3)
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            result = mission.result()
            if result.success:
                state = "success"
                message = (
                    f"Mission complete — {result.lift_peak * 100:.0f} cm lift, "
                    f"{result.max_tilt:.1f}° peak tilt"
                )
            else:
                state = "aborted" if result.aborted else "failed"
                message = result.abort_reason or "Mission did not meet the success envelope"
            self.update(
                state=state,
                message=message,
                elapsed=round(float(result.duration), 2),
                lift_cm=round(float(result.lift_peak) * 100.0, 1),
                tilt_deg=round(float(result.max_tilt), 2),
                peak_force_n=round(float(result.max_hand_force), 1),
                events=[[round(float(t), 2), text] for t, text in result.events[-4:]],
                frame_seq=self.frame_seq,
            )
        except Exception as exc:
            self.update(
                state="error",
                mode="recorded visual",
                message=f"{type(exc).__name__}: {exc}",
                frame_seq=self.frame_seq,
            )
        finally:
            if recorder is not None:
                recorder.close()


def make_handler(state: DemoState, control_token: str):
    class VoiceDemoHandler(BaseHTTPRequestHandler):
        server_version = "AlpineVoiceDemo/1.0"

        def log_message(self, format: str, *args: object) -> None:
            quiet = ("/api/status", "/api/frame", "/media/", "/assets/", "/favicon.ico")
            if self.path.startswith(quiet):
                return
            super().log_message(format, *args)

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _send_bytes(self, body: bytes, content_type: str,
                        status: int = HTTPStatus.OK) -> None:
            self._headers(status, content_type, len(body))
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, body: dict[str, Any],
                       status: int = HTTPStatus.OK) -> None:
            self._send_bytes(json.dumps(body).encode(), "application/json", status)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            total = path.stat().st_size
            range_header = self.headers.get("Range")
            start, end, status = 0, total - 1, HTTPStatus.OK
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if match and (match.group(1) or match.group(2)):
                    if match.group(1):
                        start = int(match.group(1))
                        end = int(match.group(2) or total - 1)
                    else:
                        suffix = min(int(match.group(2)), total)
                        start = total - suffix
                    end = min(end, total - 1)
                    status = HTTPStatus.PARTIAL_CONTENT
            if start < 0 or start > end or start >= total:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                try:
                    self.wfile.write(stream.read(length))
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers routinely cancel speculative media ranges.
                    pass

        def _authorised(self) -> bool:
            if not control_token:
                return True
            return self.headers.get("Authorization") == f"Bearer {control_token}"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_file(DEMO_PAGE, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(state.snapshot())
            elif path == "/api/frame.jpg":
                with state.lock:
                    frame = state.frame
                if frame is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_bytes(frame, "image/jpeg")
            elif path == "/assets/poster.png":
                self._send_file(POSTER, "image/png")
            elif path == "/media/lifting_log.mp4":
                self._send_file(RECORDED_DEMO, "video/mp4")
            elif path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/livekit/ready":
                if not self._authorised():
                    self._send_json({"ok": False, "message": "Unauthorized"},
                                    HTTPStatus.UNAUTHORIZED)
                    return
                state.update(
                    livekit_connected=True,
                    voice_source="LiveKit",
                    message="LiveKit voice agent connected — awaiting operator",
                )
                self._send_json({"ok": True, "message": "LiveKit marked ready"})
                return
            if path != "/api/command":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._authorised():
                self._send_json({"ok": False, "message": "Unauthorized"},
                                HTTPStatus.UNAUTHORIZED)
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                payload = json.loads(self.rfile.read(length) or b"{}")
                phrase = str(payload.get("phrase", ""))[:200]
            except (ValueError, TypeError, json.JSONDecodeError):
                self._send_json({"ok": False, "message": "Invalid JSON"},
                                HTTPStatus.BAD_REQUEST)
                return

            intent = classify_intent(phrase)
            if self.headers.get("X-Alpine-Voice", "").lower() == "livekit":
                state.update(livekit_connected=True, voice_source="LiveKit")
            if intent == "lift_log":
                ok, message = state.start_lift(phrase, payload_mass=12.0)
                status = HTTPStatus.ACCEPTED
            elif intent == "heavy_load":
                ok, message = state.start_lift(phrase, payload_mass=30.0)
                status = HTTPStatus.ACCEPTED
            elif intent == "wind_gust":
                ok, message = state.request_gust(phrase)
                status = HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT
            elif intent == "verglas_ice":
                ok, message = state.request_ice(phrase)
                status = HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT
            elif intent == "status":
                ok, message = state.report_telemetry(phrase)
                status = HTTPStatus.OK
            elif intent == "reset":
                ok, message = state.reset_state(phrase)
                status = HTTPStatus.OK
            elif intent == "stop":
                ok, message = state.request_stop(phrase)
                status = HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT
            else:
                current = state.snapshot()["state"]
                updates = {"heard": phrase}
                if current not in ("starting", "running"):
                    updates["message"] = "Command not recognized — no motion"
                state.update(**updates)
                ok, message = False, (
                    "Available voice commands: “lift the log”, “test heavy load”, "
                    "“add a wind gust”, “simulate verglas ice”, “status report”, or “operator stop”."
                )
                status = HTTPStatus.UNPROCESSABLE_ENTITY
            self._send_json({"ok": ok, "intent": intent, "message": message}, status)

    return VoiceDemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-controlled Alpine Lift demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="demo playback speed; use 10 for a quick smoke test")
    parser.add_argument("--open", action="store_true", help="open the console in a browser")
    parser.add_argument("--control-token", default=os.environ.get("ALPINE_CONTROL_TOKEN", ""))
    args = parser.parse_args()

    state = DemoState(speed=args.speed)
    threading.Thread(
        target=state.prepare_preview,
        name="alpine-ready-preview",
        daemon=True,
    ).start()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(state, args.control_token)
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Alpine voice demo: {url}")
    print('Command: "lift the log"  |  Safety: "operator stop"')
    if args.open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
