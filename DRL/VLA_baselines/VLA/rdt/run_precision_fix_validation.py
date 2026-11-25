"""Validate an owned corrected RDT server, then release its GPU in finally."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess

import requests

from VLA.run_evaluation_protocol import validate_protocol


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "VLA/rdt/finetune/cam_v6_h10_fp32_master_warmstart2000_20000/checkpoints/step-015000"
REPORT = ROOT / "VLA/audits/rdt_fp32_deployment_fix"
PORT = 5618


def owns_server(pid):
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return False
    args = [value.decode() for value in path.read_bytes().split(b"\0") if value]
    return ("VLA.rdt.server" in args and "--port" in args and args[args.index("--port") + 1] == str(PORT)
            and "--checkpoint" in args and Path(args[args.index("--checkpoint") + 1]).resolve() == CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owned-server-pid", required=True, type=int)
    args = parser.parse_args()
    if not owns_server(args.owned_server_pid):
        raise RuntimeError("Refusing to manage a different/missing server process")
    status_path = REPORT / "validation_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"status": "starting", "pid": os.getpid(), "owned_server_pid": args.owned_server_pid,
              "final_test_used": False, "jobs": []}
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        metadata = requests.get(f"http://127.0.0.1:{PORT}/metadata", timeout=30).json()
        if metadata.get("weight_dtype") != "float32" or metadata.get("compute") != "bfloat16_autocast":
            raise RuntimeError("Server is not serving the corrected precision profile")
        status["server_metadata"] = metadata
        protocol, protocol_hash = validate_protocol(ROOT / "VLA/evaluation_protocol.json",
            ROOT / "VLA/audits/evaluation_protocol_oracle_v3_spatial_fixed.json")
        status["protocol_sha256"] = protocol_hash
        python = str(ROOT / "VLA/.venvs/openvla/bin/python")
        source = ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"
        commands = [
            ("source_predictions", [python, "-m", "VLA.rdt.audit_checkpoint_predictions",
                "--checkpoint", str(CHECKPOINT), "--source", str(source),
                "--audit-report", str(ROOT / "VLA/audits/camera_v6_balanced_600_oneshot/audit.json"),
                "--port", str(PORT), "--episodes-per-task", "1",
                "--output", str(REPORT / "source_predictions.json")]),
        ]
        command = [python, "-m", "VLA.benchmark", "--model", "rdt", "--checkpoint", str(CHECKPOINT),
            "--port", str(PORT), "--robot", "ur5e", "--tasks", "touch,pick,move,pick_place",
            "--test-id", "precision_fix_v6_rdt15000_seed20262001_rp1", "--seed", "20262001",
            "--scene-profile", "rl_random_env", "--control-hz", "10", "--replan-steps", "1",
            "--required-success-steps", "1", "--success-mode", "semantic", "--success-threshold-scale", "1.25",
            "--audit-observation-every", "20"]
        for name, value in protocol["success"]["semantic_thresholds"].items():
            command.extend(["--semantic-" + name.replace("_", "-"), str(value)])
        commands.append(("development_rollout", command))
        for name, command in commands:
            status.update(status="running", current_job=name)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            started = datetime.now(timezone.utc).isoformat()
            with (REPORT / (name + ".log")).open("x") as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status["jobs"].append({"job": name, "started_at": started, "exit_code": completed.returncode})
            if completed.returncode:
                raise RuntimeError(f"Precision validation stage failed: {name}")
        status.update(status="checks_complete", success_improvement="Read actual rollout results; not inferred from reload equality")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if owns_server(args.owned_server_pid):
            os.kill(args.owned_server_pid, signal.SIGTERM)
            status["owned_server_stop_requested"] = True
        status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
