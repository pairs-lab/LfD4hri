"""Audit full frozen-split recovery views, then prepare TRAIN tensors only."""
from datetime import datetime, timezone
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "VLA/datasets/candidates/wrist_v8_recovery_complete_frozen_split_20261008"
AUDIT = ROOT / "VLA/audits/grasp_recovery_20261008/complete_split"
DATASET = ROOT / "VLA/datasets/lerobot_wrist_v8_recovery_train_20261008"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="initial")
    args = parser.parse_args()
    if not args.attempt.isalnum():
        raise ValueError("Attempt label must be alphanumeric")
    AUDIT.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / ("driver_status_" + args.attempt + ".json")
    if status_path.exists():
        raise FileExistsError(status_path)
    history = []
    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(), "history": history,
            "final_test_used": False, "gpu_training_started": False}, indent=2) + "\n")
    commands = []
    for split in ("train", "validation"):
        manifest = json.loads((VIEW / split / "dataset_manifest.json").read_text())
        commands.append((split + "_source", [sys.executable, "-m", "VLA.audit_finetune_data",
            "--dataset", str(VIEW / split), "--output", str(AUDIT / (split + "_source")),
            "--expected-task-counts", json.dumps(manifest["declared_task_counts"])]))
    commands.append(("native_preparation", [str(ROOT / "VLA/third_party/lerobot/.venv/bin/python"),
        "-m", "VLA.prepare_corrective_v8", "--source", str(VIEW / "train"),
        "--dataset-root", str(DATASET), "--audit-root", str(AUDIT),
        "--repo-id", "local/wrist_v8_recovery_train_20261008", "--log-prefix", "v8_complete_recovery_prepare"]))
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    try:
        for stage, cmd in commands:
            status("running", stage)
            log = ROOT / "VLA/logs" / ("v8_complete_recovery_" + args.attempt + "_" + stage + ".log")
            with log.open("x") as stream:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            history.append({"stage": stage, "exit_code": result.returncode, "log": str(log)})
            if result.returncode:
                raise RuntimeError(f"Preparation stopped at {stage}; no failed data consumed")
        status("prepared_pending_visual_review_and_native_gpu_smoke", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", stage)
        raise


if __name__ == "__main__":
    main()
