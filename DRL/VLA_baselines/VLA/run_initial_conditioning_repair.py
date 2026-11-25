"""Small sampling-repair experiment followed by conditioning and dev rollouts.

Never runs final-test seeds. Existing experiments/datasets remain untouched.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
SOURCE = ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"
AUDIT = ROOT / "VLA/audits/camera_v6_balanced_600_oneshot"
RUN = ROOT / "VLA/smolvla/finetune/cam_v6_balanced_initial30_warmstart_connector3000_1000updates"
CHECKPOINT = RUN / "checkpoints/008000/pretrained_model"
REPORT = ROOT / "VLA/audits/initial_conditioning_repair"


def commands():
    yield "train", [str(PYTHON), "-m", "VLA.smolvla.training", "train",
        "--source", str(SOURCE), "--dataset-root", str(ROOT / "VLA/datasets/lerobot_cam_v6_balanced_600_oneshot"),
        "--audit-report", str(AUDIT / "audit.json"), "--roundtrip-report", str(AUDIT / "lerobot_roundtrip.json"),
        "--stats-report", str(AUDIT / "smolvla_exact_numeric_stats.json"),
        "--preprocess-report", str(AUDIT / "smolvla_exact_worst_preprocess.json"),
        "--base-model", str(ROOT / "VLA/smolvla/finetune/cam_v6_visual_connector_fp32_warmstart1000_bs2_accum8_5000updates/checkpoints/024000/pretrained_model"),
        "--train-visual-connector", "--learning-rate", "0.00003",
        "--initial-frame-sampling-probability", "0.30", "--action-horizon", "10",
        "--batch-size", "2", "--gradient-accumulation", "8", "--steps", "1000",
        "--save-freq", "250", "--output", str(RUN)]
    yield "precision", [str(PYTHON), "-m", "VLA.smolvla.audit_training_precision",
        "--checkpoint", str(CHECKPOINT), "--output", str(REPORT / "training_precision.json")]
    yield "fingerprint", [str(PYTHON), "-m", "VLA.fingerprint_checkpoint", str(CHECKPOINT)]
    yield "conditioning", [str(PYTHON), "-m", "VLA.smolvla.audit_conditioning",
        "--checkpoint", str(CHECKPOINT), "--paired-data", str(ROOT / "VLA/audits/other_models_followup_20261007/paired_instruction_collection_pilot"),
        "--output", str(REPORT / "conditioning.json")]
    yield "development", [str(ROOT / "VLA/.venvs/openvla/bin/python"), "-m", "VLA.evaluate_v6_model",
        "--model", "smolvla", "--checkpoint", str(CHECKPOINT), "--development-only",
        "--run-tag", "v6_balanced_initial30_connector3000_new1000updates"]


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("Sampling-repair experiment already exists; refusing overwrite")
    REPORT.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1", HF_HOME=str(ROOT / "VLA/checkpoints/huggingface"),
        VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB="3100")
    status = {"status": "starting", "pid": os.getpid(), "final_test_used": False,
        "scope": "Sampling/fresh-optimizer ablation, not proven performance fix", "jobs": []}
    status_path = REPORT / "status.json"
    for name, command in commands():
        if name == "fingerprint":
            if not json.loads((REPORT / "training_precision.json").read_text())["passed"]:
                raise RuntimeError("Saved master/moment precision audit failed")
        log = REPORT / (name + ".log")
        record = {"job": name, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
        status.update(status="running", current_job=name)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
        status["jobs"].append(record)
        if result.returncode:
            status.update(status="failed", failed_job=name)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            return result.returncode
    status.update(status="development_checks_complete", performance_improvement="Read conditioning and development results; do not infer it from training completion")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
