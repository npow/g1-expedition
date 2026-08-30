"""Safety-boundary tests for the voice demo."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from voice_demo import classify_intent  # noqa: E402


class VoiceIntentTests(unittest.TestCase):
    def test_explicit_lift_commands(self) -> None:
        for phrase in (
            "lift the log",
            "please raise that trunk",
            "pick up the timber",
            "hoist the tree",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "lift_log")

    def test_stop_wins_over_lift(self) -> None:
        self.assertEqual(classify_intent("stop lifting the log"), "stop")
        self.assertEqual(classify_intent("operator stop"), "stop")

    def test_wind_gust_commands(self) -> None:
        for phrase in (
            "add a wind gust",
            "simulate wind",
            "trigger the blizzard",
            "apply a gust",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "wind_gust")

    def test_wind_noun_alone_does_not_act(self) -> None:
        self.assertIsNone(classify_intent("what is the wind doing"))

    def test_verglas_ice_commands(self) -> None:
        for phrase in (
            "simulate verglas",
            "apply ice",
            "trigger verglas underfoot",
            "simulate black ice",
            "icy terrain condition",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "verglas_ice")

    def test_heavy_load_commands(self) -> None:
        for phrase in (
            "test heavy load",
            "simulate heavy log",
            "lift thirty kilo obstacle",
            "weigh heavy load",
            "heavy load test",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "heavy_load")

    def test_status_telemetry_commands(self) -> None:
        for phrase in (
            "status report",
            "telemetry readout",
            "what is the status",
            "check status",
            "diagnostics",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "status")

    def test_reset_commands(self) -> None:
        for phrase in (
            "reset system",
            "rearm",
            "ready stance",
            "system reset",
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_intent(phrase), "reset")

    def test_negated_lift_does_not_move(self) -> None:
        self.assertIsNone(classify_intent("don't lift the log"))
        self.assertIsNone(classify_intent("do not hoist the timber"))

    def test_missing_object_or_action_does_not_move(self) -> None:
        self.assertIsNone(classify_intent("tell me about the log"))
        self.assertIsNone(classify_intent("lift the weather station"))

    def test_substrings_do_not_become_commands(self) -> None:
        self.assertIsNone(classify_intent("the forklift moved a log"))
        self.assertIsNone(classify_intent("this system is unstoppable"))


if __name__ == "__main__":
    unittest.main()

