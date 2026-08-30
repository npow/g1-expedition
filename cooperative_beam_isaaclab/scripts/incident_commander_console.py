#!/usr/bin/env python3
"""Credential-free console demo for the same core used by the voice agent."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cooperative_beam_isaaclab.tasks.incident_commander import (
    IncidentCommander,
    SupervisoryIntent,
)
from cooperative_beam_isaaclab.tasks.livekit_state_bus import RobotCoordinationState


class DemoTelemetry:
    """Clearly labelled, deterministic telemetry for laptop judging demos."""

    def __init__(self, commander: IncidentCommander) -> None:
        self.commander = commander
        self.sequence = 0
        count = len(commander.expected_robot_ids)
        self.loads = [0.5 if count == 3 and index == 1 else 0.5 / (count - 1) for index in range(count)]
        if count != 3:
            self.loads = [1.0 / count] * count

    def refresh(self) -> None:
        now = time.monotonic_ns()
        for index, (robot_id, load) in enumerate(zip(self.commander.expected_robot_ids, self.loads, strict=True)):
            self.commander.ingest_state(
                RobotCoordinationState(
                    robot_id=robot_id,
                    sequence=self.sequence,
                    sender_time_ns=time.time_ns(),
                    position_w=(0.35 * index, (-1.0) ** index * 0.45, 0.82),
                    linear_velocity_w=(0.08, 0.0, 0.0),
                    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                    load_ratio=load,
                ),
                received_monotonic_ns=now,
            )
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF


def robot_id_from_text(value: str) -> str | None:
    match = re.search(r"\bg\s*1[\s_-]*(\d+)\b", value, re.IGNORECASE)
    return None if match is None else f"g1_{match.group(1)}"


def dispatch(commander: IncidentCommander, demo: DemoTelemetry, value: str) -> str:
    text = value.strip().lower()
    demo.refresh()
    robot_id = robot_id_from_text(text)
    if not text:
        return ""
    if text in {"help", "?"}:
        return (
            "Try: report status; assess load; report G1-1; begin lift; confirm start; hold; "
            "resume; confirm resume; simulate loss of G1-1; why did you stop; restore G1-1; abort."
        )
    if "restore" in text and robot_id:
        return commander.restore_link(robot_id)
    if ("simulate" in text or "inject" in text) and ("loss" in text or "drop" in text) and robot_id:
        return commander.inject_link_loss(robot_id)
    if "why" in text or "explain" in text:
        return commander.explain_last_transition()
    if robot_id and ("load" in text or "report" in text or "status" in text):
        return commander.robot_load(robot_id)
    if "assess" in text or ("load" in text and "status" not in text):
        return commander.assess_load()
    if "status" in text or "report" in text:
        return commander.status()
    if text in {"begin", "begin lift", "start", "start lift"}:
        return commander.request_command(SupervisoryIntent.START).message
    if text == "confirm start":
        return commander.request_command(SupervisoryIntent.START, confirmed=True).message
    if text in {"hold", "hold position"}:
        return commander.request_command(SupervisoryIntent.HOLD).message
    if text == "resume":
        return commander.request_command(SupervisoryIntent.RESUME).message
    if text == "confirm resume":
        return commander.request_command(SupervisoryIntent.RESUME, confirmed=True).message
    if text in {"abort", "abort mission", "stop"}:
        return commander.request_command(SupervisoryIntent.ABORT).message
    return "Command not recognized. Say help for the bounded incident-command vocabulary."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G1 incident commander without API credentials")
    parser.add_argument("--team", default="g1_0,g1_1,g1_2")
    parser.add_argument("--command", action="append", default=[], help="Run a command non-interactively; repeatable")
    args = parser.parse_args()
    team = tuple(item.strip() for item in args.team.split(",") if item.strip())
    commander = IncidentCommander(team, demo_mode=True)
    demo = DemoTelemetry(commander)
    demo.refresh()
    print("G1 Expedition Incident Commander — DEMO TELEMETRY, not a live robot session")
    print("Commands are supervisory only; emergency stop remains local and physical.")
    commands = args.command
    if commands:
        for command in commands:
            print(f"> {command}")
            print(dispatch(commander, demo, command))
        return
    print(dispatch(commander, demo, "help"))
    while True:
        try:
            command = input("expedition> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command.strip().lower() in {"quit", "exit"}:
            break
        response = dispatch(commander, demo, command)
        if response:
            print(response)


if __name__ == "__main__":
    main()
