"""Bounded tiny-train fit and learned source replay; no benchmark score."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "VLA/audits/finetune_v8_tiny_fit_20261008"
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_tiny_fit_8_20261008"
DATASET = ROOT / "VLA/datasets/lerobot_wrist_v8_tiny_fit_8_20261008"
OUTPUT = ROOT / "VLA/smolvla/finetune/v8_tiny8_vlm_fp32_u500"
BASE = ROOT / "VLA/smolvla/finetune/cam_v8_grouped_vlm_targetbalance_warm_u1000/checkpoints/016000/pretrained_model"
CP = OUTPUT / "checkpoints/004000/pretrained_model"


def main():
    status_path = AUDIT / "driver_status.json"
    if status_path.exists() or OUTPUT.exists():
        raise FileExistsError("Do not overwrite a tiny-fit trial")
    history = []
    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(),
            "history": history, "scope": "Training-set memorization diagnostic only", "final_test_used": False,
            "updates": 500, "samples": 4000, "source_frames": 171}, indent=2) + "\n")
    while not (AUDIT / "roundtrip.json").exists():
        status("waiting_for_data", "validate")
        time.sleep(5)
    for path in [AUDIT / "source/audit.json", AUDIT / "roundtrip.json", AUDIT / "stats.json", AUDIT / "smolvla_preprocess.json"]:
        if not json.loads(path.read_text()).get("passed"):
            status("failed_data_gate", "validate")
            raise RuntimeError(f"Failed tiny source gate: {path}")
    common = ["--source", str(SOURCE), "--dataset-root", str(DATASET),
        "--audit-report", str(AUDIT / "source/audit.json"), "--roundtrip-report", str(AUDIT / "roundtrip.json"),
        "--stats-report", str(AUDIT / "stats.json"), "--preprocess-report", str(AUDIT / "smolvla_preprocess.json"),
        "--base-model", str(BASE), "--train-vlm", "--train-visual-connector",
        "--steps", "500", "--batch-size", "1", "--gradient-accumulation", "8",
        "--learning-rate", "0.0001", "--warmup-updates", "50", "--decay-updates", "1000",
        "--initial-frame-sampling-probability", "0.3", "--target-balanced-sampling", "--action-horizon", "10"]
    stages = [
        ("validate", ["-m", "VLA.smolvla.training", "validate", *common]),
        ("train", ["-m", "VLA.smolvla.training", "train", *common, "--save-freq", "500", "--output", str(OUTPUT)]),
        ("fingerprint", ["-m", "VLA.fingerprint_checkpoint", str(CP)]),
        ("input_parity", ["-m", "VLA.smolvla.audit_train_infer_inputs", "--checkpoint", str(CP),
            "--source", str(SOURCE), "--audit-report", str(AUDIT / "source/audit.json"),
            "--episodes-per-task", "2", "--output", str(AUDIT / "input_parity.json")]),
        ("imitation", ["-m", "VLA.smolvla.audit_checkpoint_predictions", "--checkpoint", str(CP),
            "--source", str(SOURCE), "--audit-report", str(AUDIT / "source/audit.json"),
            "--episodes-per-task", "2", "--output", str(AUDIT / "imitation.json")]),
        ("learned_replay", ["-m", "VLA.evaluate_tiny_fit", "--checkpoint", str(CP), "--source", str(SOURCE),
            "--audit-report", str(AUDIT / "source/audit.json"), "--output", str(AUDIT / "learned_replay")]),
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "TOKENIZERS_PARALLELISM": "false"})
    for name, command in stages:
        if name in ("train", "imitation", "learned_replay"):
            while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                status("waiting_for_gpu", name)
                time.sleep(5)
        path = ROOT / "VLA/logs" / ("v8_tiny8_fit_" + name + ".log")
        if path.exists():
            raise FileExistsError(path)
        status("running", name)
        with path.open("w") as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"stage": name, "exit_code": result.returncode, "log": str(path)})
        if result.returncode:
            status("failed", name)
            raise RuntimeError(f"Tiny fit failed at {name}: {path}")
    status("completed_train_diagnostic_only", "done")


if __name__ == "__main__":
    main()
