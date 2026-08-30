#!/usr/bin/env python3
"""LiveKit coordination bridge for one deployed G1 policy process.

Input is JSON Lines on stdin. Each line must contain the robot's world-frame
position, velocity, orientation, and measured team load ratio. When a
98-value ``local_observation`` is included, output contains the complete
checkpoint-compatible MAPPO actor observation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cooperative_beam_isaaclab.tasks.livekit_state_bus import (
    LiveKitRobotStateClient,
    assemble_policy_observation,
    teammate_observation_parts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fan out compact G1 coordination state through a LiveKit room.")
    parser.add_argument("--robot-id", required=True, help="Canonical policy identity, for example g1_0")
    parser.add_argument("--team", required=True, help="Comma-separated room identities, for example g1_0,g1_1,g1_2")
    parser.add_argument("--url", default=os.environ.get("LIVEKIT_URL"))
    parser.add_argument("--token", default=os.environ.get("LIVEKIT_TOKEN"))
    parser.add_argument("--max-age-ms", type=float, default=150.0)
    parser.add_argument("--settle-ms", type=float, default=0.0, help="Optional wait after publish before snapshot")
    parser.add_argument(
        "--command-caller",
        action="append",
        default=[],
        help="LiveKit identity allowed to enqueue supervisory RPC commands; repeat as needed",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    if not args.url or not args.token:
        raise SystemExit("Set LIVEKIT_URL and a per-robot LIVEKIT_TOKEN, or pass --url/--token")
    team = [item.strip() for item in args.team.split(",") if item.strip()]
    if args.robot_id not in team:
        raise SystemExit(f"--robot-id {args.robot_id!r} is not present in --team")
    teammate_ids = [robot_id for robot_id in team if robot_id != args.robot_id]
    client = LiveKitRobotStateClient(
        args.robot_id,
        args.url,
        args.token,
        command_callers=args.command_caller,
    )
    await client.connect()
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            payload = json.loads(line)
            local_state = client.publish(
                payload["position_w"],
                payload["linear_velocity_w"],
                payload["orientation_wxyz"],
                payload["load_ratio"],
            )
            if args.settle_ms > 0:
                await asyncio.sleep(args.settle_ms / 1000.0)
            if "local_observation" in payload:
                observation, fresh = assemble_policy_observation(
                    payload["local_observation"],
                    local_state,
                    teammate_ids,
                    client.table,
                    max_age_ms=args.max_age_ms,
                )
                result = {"policy_observation": observation, "teammate_fresh": fresh}
            else:
                kinematics, loads, fresh = teammate_observation_parts(
                    local_state,
                    teammate_ids,
                    client.table,
                    max_age_ms=args.max_age_ms,
                )
                result = {
                    "teammate_kinematics": kinematics,
                    "teammate_loads": loads,
                    "teammate_fresh": fresh,
                }
            # The bridge never applies commands itself. It gives the local,
            # deterministic supervisor a bounded queue to admit or reject.
            result["supervisory_commands"] = client.drain_commands()
            print(json.dumps(result, separators=(",", ":")), flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
