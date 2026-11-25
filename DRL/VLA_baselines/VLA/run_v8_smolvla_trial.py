"""Fail-closed bounded SmolVLA trial and automatic DEVELOPMENT follow-up.

No final seeds. A source/loader/numeric/preprocessor gate and real GPU smoke
must pass before the bounded train stage. Task success is never inferred from
smoke, teacher replay, training loss or diagnostic imitation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_grouped_train_20261008"
VALIDATION = ROOT / "VLA/datasets/candidates/wrist_v8_grouped_validation_20261008"
DATASET = ROOT / "VLA/datasets/lerobot_wrist_v8_grouped_train_20261008"
AUDIT = ROOT / "VLA/audits/finetune_v8_20261008"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--adapt-vlm", action="store_true")
    parser.add_argument("--target-balanced", action="store_true")
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    args = parser.parse_args()
    if args.updates < 20 or args.updates > 1000:
        raise ValueError("This diagnostic driver permits only 20..1000 optimizer updates")
    batch, accumulation = (1, 16) if args.adapt_vlm else (2, 8)
    recipe = "vlm_targetbalance_warm" if args.adapt_vlm and args.target_balanced else "connector_fp32_bs2_accum8"
    if args.target_balanced and not args.adapt_vlm:
        recipe = "connector_targetbalance"
    elif args.adapt_vlm and not args.target_balanced:
        recipe = "vlm_warm"
    tag = f"cam_v8_grouped_{recipe}_u{args.updates}"
    smoke_root = ROOT / ("VLA/smolvla/finetune/cam_v8_grouped_" + recipe + "_smoke")
    train_root = ROOT / "VLA/smolvla/finetune" / tag
    smoke_cp = smoke_root / "checkpoints" / f"{2 * accumulation:06d}" / "pretrained_model"
    checkpoint = train_root / "checkpoints" / f"{args.updates * accumulation:06d}" / "pretrained_model"
    followup = AUDIT / tag
    common = ["--source", str(SOURCE), "--dataset-root", str(DATASET),
              "--audit-report", str(AUDIT / "train_source/audit.json"),
              "--roundtrip-report", str(AUDIT / "lerobot_roundtrip.json"),
              "--stats-report", str(AUDIT / "smolvla_exact_numeric_stats.json"),
              "--preprocess-report", str(AUDIT / "smolvla_exact_worst_preprocess.json"),
              "--steps", str(args.updates), "--batch-size", str(batch), "--gradient-accumulation", str(accumulation),
              "--base-model", args.base_model,
              "--learning-rate", "0.0001", "--action-horizon", "10", "--train-visual-connector",
              "--initial-frame-sampling-probability", "0.3"]
    if args.adapt_vlm:
        common.append("--train-vlm")
    if args.target_balanced:
        common.append("--target-balanced-sampling")
    commands = [
        ("validate", ["-m", "VLA.smolvla.training", "validate", *common]),
        ("gpu_smoke", ["-m", "VLA.smolvla.training", "smoke", *common, "--output", str(smoke_root)]),
        ("smoke_fingerprint", ["-m", "VLA.fingerprint_checkpoint", str(smoke_cp)]),
        ("smoke_input_parity", ["-m", "VLA.smolvla.audit_train_infer_inputs", "--checkpoint", str(smoke_cp),
            "--source", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
            "--episodes-per-task", "2", "--output", str(followup / "smoke_train_infer_parity.json")]),
        ("bounded_train", ["-m", "VLA.smolvla.training", "train", *common,
            "--warmup-updates", str(args.updates // 10), "--decay-updates", str(args.updates),
            "--save-freq", str(args.updates), "--output", str(train_root)]),
        ("fingerprint", ["-m", "VLA.fingerprint_checkpoint", str(checkpoint)]),
        ("train_imitation", ["-m", "VLA.smolvla.audit_checkpoint_predictions", "--checkpoint", str(checkpoint),
            "--source", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
            "--episodes-per-task", "6", "--output", str(followup / "train_imitation.json")]),
        ("heldout_imitation", ["-m", "VLA.smolvla.audit_checkpoint_predictions", "--checkpoint", str(checkpoint),
            "--source", str(VALIDATION), "--audit-report", str(AUDIT / "validation_source/audit.json"),
            "--normalization-source", str(SOURCE), "--normalization-audit-report", str(AUDIT / "train_source/audit.json"),
            "--episodes-per-task", "6", "--output", str(followup / "heldout_imitation.json")]),
        ("text_conditioning", ["-m", "VLA.smolvla.audit_conditioning", "--checkpoint", str(checkpoint),
            "--paired-data", str(ROOT / "VLA/datasets/candidates/paired_wrist_v8_scene_grasp_64_allvocab"),
            "--layouts-per-task", "2", "--output", str(followup / "chunk_conditioning.json")]),
        ("development", ["-m", "VLA.run_evaluation_protocol", "--model", "smolvla", "--split", "development",
            "--replan-steps", "1", "--checkpoint", str(checkpoint), "--protocol", str(ROOT / "VLA/evaluation_protocol_v8.json"),
            "--oracle-audit", str(AUDIT / "evaluation_protocol_development_oracle.json"), "--run-tag", tag]),
        ("development_video_repeat", ["-m", "VLA.benchmark", "--model", "smolvla", "--checkpoint", str(checkpoint),
            "--scene-profile", "rl_random_env", "--seed", "20262001", "--replan-steps", "1",
            "--required-success-steps", "1", "--success-mode", "semantic", "--audit-observation-every", "10",
            "--test-id", tag + "_dev_seed20262001_video_repeat"]),
    ]
    if not args.run:
        print(json.dumps({"canary_only": True, "final_test_used": False,
                          "commands": [{"stage": name, "command": [sys.executable, *cmd]} for name, cmd in commands]}, indent=2))
        return
    status_path = AUDIT / (tag + "_driver_status.json")
    if status_path.exists() or smoke_root.exists() or train_root.exists():
        raise FileExistsError("Immutable trial already exists; do not overwrite or silently resume")
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
                        "TOKENIZERS_PARALLELISM": "false", "HF_DATASETS_DISABLE_PROGRESS_BARS": "1"})
    history = []
    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(), "canary_only": True,
            "final_test_used": False, "planned_optimizer_updates": args.updates, "history": history}, indent=2) + "\n")
    gates = [AUDIT / name for name in ("train_source/audit.json", "validation_source/audit.json",
              "lerobot_roundtrip.json", "smolvla_exact_numeric_stats.json", "smolvla_exact_worst_preprocess.json")]
    deadline = time.monotonic() + 1800
    while not all(path.is_file() for path in gates):
        status("waiting_for_data_audits", "validate")
        if time.monotonic() > deadline:
            raise TimeoutError("Pending audits did not complete; no training was started")
        time.sleep(5)
    if any(not json.loads(path.read_text()).get("passed") for path in gates):
        status("failed_data_gate", "validate")
        raise RuntimeError("A required data/pipeline audit failed")
    for name, command in commands:
        while name not in ("validate", "smoke_fingerprint", "smoke_input_parity", "fingerprint"):
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if not active:
                break
            status("waiting_for_gpu", name)
            time.sleep(5)
        log_path = ROOT / "VLA/logs" / (tag + "_" + name + ".log")
        if log_path.exists():
            raise FileExistsError(log_path)
        status("running", name)
        print(f"Starting {name}: {log_path}", flush=True)
        with log_path.open("w") as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"stage": name, "exit_code": result.returncode, "log": str(log_path)})
        if result.returncode:
            status("failed", name)
            raise RuntimeError(f"{name} failed; inspect {log_path} before continuing")
    status("completed_diagnostic_only", "done")


if __name__ == "__main__":
    main()
