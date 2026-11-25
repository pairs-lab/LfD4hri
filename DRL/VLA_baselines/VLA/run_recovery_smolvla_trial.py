"""Qualified recovery-source trial, serialized after the original pi0 workflow."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.evaluate_openpi_development import process_identity

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "VLA/audits/grasp_recovery_20261008/complete_split"
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_recovery_complete_frozen_split_20261008/train"
VALIDATION = SOURCE.parent / "validation"
DATASET = ROOT / "VLA/datasets/lerobot_wrist_v8_recovery_train_20261008"
BASE = ROOT / "VLA/smolvla/finetune/cam_v8_grouped_vlm_targetbalance_warm_u1000/checkpoints/016000/pretrained_model"
TAG = "cam_v8_recovery_vlm_fp32_u8000"
OUTPUT = ROOT / "VLA/smolvla/finetune" / TAG
SMOKE = ROOT / "VLA/smolvla/finetune" / (TAG + "_smoke2")
CP = OUTPUT / "checkpoints/064000/pretrained_model"
SMOKE_CP = SMOKE / "checkpoints/000016/pretrained_model"


def main():
    status_path = AUDIT / "learned_trial_status.json"
    if status_path.exists() or OUTPUT.exists() or SMOKE.exists():
        raise FileExistsError("Immutable trial already exists")
    geometry_gate = ROOT / "VLA/audits/oracle_grasp_quality_20261008/TRAINING_GATE.json"
    if not geometry_gate.exists() or not json.loads(geometry_gate.read_text()).get("training_on_affected_demonstrations_allowed"):
        raise RuntimeError("Oracle grasp geometry is not qualified; do not train solely because conversion audits pass")
    review = json.loads((AUDIT / "visual_review.json").read_text())
    if (not review.get("passed") or review["source_index_sha256"] != sha256_file(SOURCE / "index.json")
            or review["preprocess_report_sha256"] != sha256_file(AUDIT / "smolvla_exact_worst_preprocess.json")):
        raise RuntimeError("Missing/stale actual image review")
    for path in ("train_source/audit.json", "validation_source/audit.json", "lerobot_roundtrip.json", "smolvla_exact_numeric_stats.json"):
        if not json.loads((AUDIT / path).read_text()).get("passed"):
            raise RuntimeError(f"Failed native data gate: {path}")
    history = []
    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(),
            "process_start_identity": process_identity(os.getpid()), "updated_at": datetime.now(timezone.utc).isoformat(),
            "history": history, "planned_optimizer_updates": 8000, "final_test_used": False,
            "scope": "New-source development trial; no stable-performance claim"}, indent=2) + "\n")
    baseline_pid = 1622003
    baseline_identity = process_identity(baseline_pid)
    common = ["--source", str(SOURCE), "--dataset-root", str(DATASET), "--repo-id", "local/wrist_v8_recovery_train_20261008",
        "--audit-report", str(AUDIT / "train_source/audit.json"), "--roundtrip-report", str(AUDIT / "lerobot_roundtrip.json"),
        "--stats-report", str(AUDIT / "smolvla_exact_numeric_stats.json"),
        "--preprocess-report", str(AUDIT / "smolvla_exact_worst_preprocess.json"), "--base-model", str(BASE),
        "--train-vlm", "--train-visual-connector", "--steps", "8000", "--batch-size", "1", "--gradient-accumulation", "8",
        "--learning-rate", "0.0001", "--initial-frame-sampling-probability", "0.3", "--target-balanced-sampling", "--action-horizon", "10"]
    commands = [
        ("validate", ["-m", "VLA.smolvla.training", "validate", *common]),
        ("gpu_smoke", ["-m", "VLA.smolvla.training", "smoke", *common, "--output", str(SMOKE)]),
        ("smoke_fingerprint", ["-m", "VLA.fingerprint_checkpoint", str(SMOKE_CP)]),
        ("smoke_input_parity", ["-m", "VLA.smolvla.audit_train_infer_inputs", "--checkpoint", str(SMOKE_CP),
            "--source", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"), "--output", str(AUDIT / "smoke_input_parity.json")]),
        ("train", ["-m", "VLA.smolvla.training", "train", *common, "--warmup-updates", "800", "--decay-updates", "8000",
            "--save-freq", "2000", "--output", str(OUTPUT)]),
        ("fingerprint", ["-m", "VLA.fingerprint_checkpoint", str(CP)]),
        ("heldout_imitation", ["-m", "VLA.smolvla.audit_checkpoint_predictions", "--checkpoint", str(CP),
            "--source", str(VALIDATION), "--audit-report", str(AUDIT / "validation_source/audit.json"),
            "--normalization-source", str(SOURCE), "--normalization-audit-report", str(AUDIT / "train_source/audit.json"),
            "--episodes-per-task", "6", "--output", str(AUDIT / "heldout_imitation.json")]),
        ("heldout_recovery_imitation", ["-m", "VLA.smolvla.audit_checkpoint_predictions", "--checkpoint", str(CP),
            "--source", str(VALIDATION), "--audit-report", str(AUDIT / "validation_source/audit.json"),
            "--normalization-source", str(SOURCE), "--normalization-audit-report", str(AUDIT / "train_source/audit.json"),
            "--episodes-per-task", "6", "--file-prefix", "corrective_", "--output", str(AUDIT / "heldout_recovery_imitation.json")]),
        ("development", ["-m", "VLA.run_evaluation_protocol", "--model", "smolvla", "--checkpoint", str(CP),
            "--protocol", str(ROOT / "VLA/evaluation_protocol_v8.json"),
            "--oracle-audit", str(ROOT / "VLA/audits/finetune_v8_20261008/evaluation_protocol_development_oracle.json"),
            "--split", "development", "--replan-steps", "1", "--run-tag", TAG]),
        ("video_repeat", ["-m", "VLA.benchmark", "--model", "smolvla", "--checkpoint", str(CP),
            "--scene-profile", "rl_random_env", "--seed", "20262001", "--replan-steps", "1", "--required-success-steps", "1",
            "--success-mode", "semantic", "--audit-observation-every", "10", "--test-id", TAG + "_dev_seed20262001_video_repeat"]),
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "TOKENIZERS_PARALLELISM": "false"})
    deadline = time.monotonic() + 24 * 3600
    try:
        for stage, command in commands:
            if stage != "validate":
                while True:
                    baseline_alive = baseline_identity is not None and process_identity(baseline_pid) == baseline_identity
                    gpu_busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                    if not baseline_alive and not gpu_busy:
                        break
                    status("waiting_for_original_pi0_workflow_and_gpu", stage)
                    if time.monotonic() > deadline:
                        raise TimeoutError("Queue deadline exceeded; no competing model launched")
                    time.sleep(15)
            status("running", stage)
            log = ROOT / "VLA/logs" / (TAG + "_" + stage + ".log")
            with log.open("x") as stream:
                child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        status("running", stage)
            history.append({"stage": stage, "exit_code": code, "log": str(log)})
            if code:
                raise RuntimeError(f"Stage {stage} failed; no oracle/score substitution")
        status("development_completed_pending_video_and_failure_review", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", stage)
        raise


if __name__ == "__main__":
    main()
