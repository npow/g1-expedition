"""Small visual smoke test for the selected learned policy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from record_demo import record_policy_demo


if __name__ == "__main__":
    report = record_policy_demo(
        model_path="models/ppo_self_arrest/g1_self_arrest_final.zip",
        output_video="videos/test_learned_arrest.mp4",
        seed=2026,
    )
    if not report["success"]:
        raise SystemExit("learned-policy visual smoke test did not arrest")
