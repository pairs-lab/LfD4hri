"""Resume the unmodified V8 baseline and evaluate only after actual completion."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "VLA/audits/finetune_v8_20261008"
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_grouped_train_20261008"
EXPERIMENT = "cam_v8_h10_colour_safe_bs1_accum4_10000updates"
CHECKPOINT = ROOT / "VLA/pi0/finetune/pi0_pybullet_ur5e_robotiq" / EXPERIMENT / "39999"


def main():
    output = ROOT / "VLA/audits/grasp_recovery_20261008/pi0_baseline_resume_status.json"
    if output.exists():
        raise FileExistsError(output)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("Do not start a second GPU compute job")
    common = ["--model", "pi0", "--repo-id", "openpi_wrist_v8_grouped_train_20261008",
        "--source-dataset", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
        "--roundtrip-report", str(AUDIT / "openpi_roundtrip.json"),
        "--action-adapter-report", str(AUDIT / "pi0_action_adapter_train.json"),
        "--closed-loop-adapter-report", str(AUDIT / "pi0_closed_loop_semantic_train.json")]
    train = [str(ROOT / "VLA/third_party/openpi/.venv/bin/python"), "-m", "VLA.pi0.training", "train", *common,
        "--action-horizon", "10", "--steps", "10000", "--batch-size", "1", "--gradient-accumulation", "4",
        "--save-interval", "1000", "--exp-name", EXPERIMENT, "--color-safe-augmentation",
        "--augmentation-report", str(AUDIT / "pi0_colour_safe_augmentation.json"), "--resume"]
    evaluation = [sys.executable, "-m", "VLA.evaluate_openpi_development", *common,
        "--checkpoint", str(CHECKPOINT), "--oracle-audit", str(AUDIT / "evaluation_protocol_development_oracle.json"),
        "--run-tag", "cam_v8_pi0_colour_safe_u10000_resumed", "--replan-steps", "1"]
    history = []
    def status(state, stage, child=None):
        output.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(), "child_pid": child,
            "updated_at": datetime.now(timezone.utc).isoformat(), "history": history,
            "source_changed": False, "final_test_used": False,
            "scope": "Original-source baseline continuation; NOT training on the new recovery dataset"}, indent=2) + "\n")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "OPENPI_DATA_HOME": str(ROOT / "VLA/cache/openpi"),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85", "PYTHONUNBUFFERED": "1"})
    try:
        for stage, command in (("resume_training", train), ("development", evaluation)):
            log = ROOT / "VLA/logs" / ("pi0_v8_baseline_resumed_" + stage + ".log")
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status("running", stage, child.pid)
                code = child.wait()
            history.append({"stage": stage, "exit_code": code, "log": str(log)})
            if code:
                raise RuntimeError(f"{stage} failed; no score substituted for integration failure")
        status("baseline_development_completed_pending_visual_review", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", stage)
        raise


if __name__ == "__main__":
    main()
