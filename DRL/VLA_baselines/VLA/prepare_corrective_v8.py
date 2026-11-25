"""Prepare the new corrective TRAIN view; never reuse old numeric statistics."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_corrective_frozen_split_20261008/train"
DATASET = ROOT / "VLA/datasets/lerobot_wrist_v8_corrective_train_20261008"
AUDIT = ROOT / "VLA/audits/finetune_v8_corrective_20261008"
REPO = "local/wrist_v8_corrective_train_20261008"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument("--repo-id", default=REPO)
    parser.add_argument("--log-prefix", default="v8_corrective_prepare")
    args = parser.parse_args()
    source, dataset, audit = args.source.resolve(), args.dataset_root.resolve(), args.audit_root.resolve()
    if Path(args.log_prefix).name != args.log_prefix or not args.log_prefix:
        raise ValueError("Log prefix must be one path component")
    status_path = audit / "preparation_status.json"
    if status_path.exists() or dataset.exists():
        raise FileExistsError("Immutable preparation already exists; choose a new lineage")
    source_audit = audit / "train_source/audit.json"
    if not json.loads(source_audit.read_text()).get("passed"):
        raise RuntimeError("Source audit failed; no conversion allowed")
    history = []

    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage,
            "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "dataset": str(dataset), "history": history,
            "final_test_used": False, "gpu_training_started": False}, indent=2) + "\n")

    stages = [
        ("convert", ["-m", "VLA.lerobot.convert_dataset", "--source", str(source),
            "--audit-report", str(source_audit), "--output", str(dataset), "--repo-id", args.repo_id]),
        ("fresh_train_stats", ["-m", "VLA.lerobot.recompute_exact_stats", "--source", str(source),
            "--dataset-root", str(dataset), "--output", str(audit / "smolvla_exact_numeric_stats.json")]),
        ("roundtrip", ["-m", "VLA.lerobot.audit_converted_dataset", "--source", str(source),
            "--dataset-root", str(dataset), "--repo-id", args.repo_id, "--output", str(audit / "lerobot_roundtrip.json")]),
        ("model_image_export", ["-m", "VLA.audit_model_preprocessing", "--model", "smolvla",
            "--dataset", str(source), "--output", str(audit / "smolvla_exact_worst_preprocess.png")]),
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        for stage, command in stages:
            log = ROOT / "VLA/logs" / (args.log_prefix + "_" + stage + ".log")
            if log.exists():
                raise FileExistsError(log)
            status("running", stage)
            with log.open("x") as stream:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT)
            history.append({"stage": stage, "exit_code": result.returncode, "log": str(log)})
            if result.returncode:
                raise RuntimeError(f"Preparation failed at {stage}; inspect {log}")
        for name in ("smolvla_exact_numeric_stats.json", "lerobot_roundtrip.json", "smolvla_exact_worst_preprocess.json"):
            if not json.loads((audit / name).read_text()).get("passed"):
                raise RuntimeError(f"Post-preparation audit failed: {name}")
        status("prepared_pending_human_image_review_and_native_training_validation", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", stage if "stage" in locals() else "setup")
        raise


if __name__ == "__main__":
    main()
