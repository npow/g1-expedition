"""Run the realism-focused fixed-line experiment on Hugging Face Jobs."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE = Path("/workspace")
WORKDIR = Path("/tmp/himalaya-fixed-line")
OUTPUT = Path("/outputs/slope_grounded_run")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("Expected the project volume at /workspace")
    shutil.copytree(SOURCE, WORKDIR, dirs_exist_ok=True)
    os.chdir(WORKDIR)
    sys.path.insert(0, str(WORKDIR))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from evaluate_fixed_line import evaluate
    from train_fixed_line import train

    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    model_path = train(
        total_timesteps=80_000,
        num_envs=24,
        save_dir=str(OUTPUT),
        seed=31,
        device="cpu",
        resume=str(WORKDIR / "resume.zip"),
    )
    report_path = OUTPUT / "evaluation_report_cloud.json"
    report = evaluate(
        str(model_path),
        episodes=4,
        seed=91_000,
        output=str(report_path),
    )

    metadata_path = OUTPUT / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "cloud_compute_used": True,
            "cloud_provider": "Hugging Face Jobs",
            "hardware_flavor": "cpu-performance",
            "job_started_utc": started.isoformat(),
            "job_finished_utc": datetime.now(timezone.utc).isoformat(),
            "cloud_evaluation_all_gates_passed": report[
                "all_verification_gates_passed"
            ],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"HF_JOB_RESULT={model_path}")
    print(f"HF_JOB_ALL_GATES_PASSED={report['all_verification_gates_passed']}")


if __name__ == "__main__":
    main()
