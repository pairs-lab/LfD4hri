#!/usr/bin/env python3
"""Select replanning on development seeds, then run the locked final split.

Use only on the quiet GPU after training. The script saves the chosen setting
before opening the final split; integration errors abort the evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "VLA/audits/camera_v6_balanced_600_oneshot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi0_fast", "openvla", "rdt", "smolvla"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--development-only", action="store_true",
                       help="Run/select on development without opening final seeds; use during setup/finetune audits")
    stage.add_argument("--final-only", action="store_true",
                       help="Run the already frozen selection without rerunning or tuning on development")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if args.model == "smolvla":
        contract_path = checkpoint / "fp32_master_contract.json"
        if not contract_path.is_file():
            raise RuntimeError("SmolVLA final comparison requires audited FP32 masters; historical BF16 runs are diagnostic only")
        contract = json.loads(contract_path.read_text())
        if not contract.get("passed") or not contract.get("saved_weights_and_adam_moments_fp32"):
            raise RuntimeError("SmolVLA FP32 master audit did not pass")
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    environment.update({
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "HF_HUB_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(ROOT / "VLA/cache/openpi"),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    })
    if not (checkpoint / "benchmark_fingerprint.json").is_file():
        subprocess.run([sys.executable, "-m", "VLA.fingerprint_checkpoint", str(checkpoint)],
                       cwd=ROOT, env=environment, check=True)
    from VLA.fingerprint_checkpoint import verify_fingerprint
    identity = verify_fingerprint(checkpoint)

    simulator_python = (
        ROOT / "VLA/third_party/lerobot/.venv/bin/python" if args.model == "smolvla"
        else ROOT / "VLA/.venvs/openvla/bin/python" if args.model == "openvla"
        else Path(sys.executable)
    )
    server_command = None
    port = 5615 if args.model == "rdt" else 8015
    if args.model == "rdt":
        server_command = [
            str(ROOT / "VLA/.venvs/rdt/bin/python"), "-m", "VLA.rdt.server",
            "--checkpoint", str(checkpoint), "--embeddings", str(ROOT / "VLA/rdt/prompt_embeddings_v6.pt"),
            "--vision", str(ROOT / "VLA/checkpoints/siglip-so400m-patch14-384"), "--port", str(port),
        ]
    elif args.model in ("pi0", "pi0_fast"):
        server_command = [
            str(ROOT / "VLA/third_party/openpi/.venv/bin/python"), "-m", "VLA.pi0.server",
            "--model", args.model, "--checkpoint", str(checkpoint),
            "--repo-id", "openpi_lerobot_cam_v6_balanced_600_oneshot",
            "--source-dataset", str(ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"),
            "--audit-report", str(AUDIT / "audit.json"), "--roundtrip-report", str(AUDIT / "openpi_roundtrip.json"),
            "--action-adapter-report", str(AUDIT / "pi0_action_adapter.json"),
            "--closed-loop-adapter-report", str(AUDIT / "pi0_closed_loop_replay.json"), "--port", str(port),
        ]
    server = None
    server_log = None
    try:
        if server_command is not None:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    raise RuntimeError(f"Dedicated evaluation port {port} is already occupied")
            except (ConnectionRefusedError, TimeoutError):
                pass
            log_path = ROOT / "VLA/logs" / f"{args.model}_{args.run_tag}_eval_server.log"
            if log_path.exists():
                raise FileExistsError(log_path)
            server_log = log_path.open("w")
            server = subprocess.Popen(server_command, cwd=ROOT, env=environment,
                                      stdout=server_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Evaluation server failed; see {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Evaluation server readiness timed out; see {log_path}")
                    time.sleep(1)

        common = [str(simulator_python), "-m", "VLA.run_evaluation_protocol", "--model", args.model,
                  "--checkpoint", str(checkpoint), "--run-tag", args.run_tag]
        if server_command is not None:
            common += ["--port", str(port)]
        if not args.final_only:
            subprocess.run([*common, "--split", "development"], cwd=ROOT, env=environment, check=True)
        dev_path = ROOT / "VLA/evaluations" / args.model / "development" / args.run_tag / "aggregate.json"
        dev = json.loads(dev_path.read_text())
        selected = min(dev["summaries"], key=lambda s: (
            -s["success_rate"], s["mean_normalized_minimum_semantic_gap"], s["replan_steps"]
        ))
        selection_path = dev_path.parent / "frozen_selection.json"
        selection = {
            "model": args.model, "checkpoint": str(checkpoint),
            "checkpoint_content_sha256": identity["content_tree_sha256"],
            "selected_replan_steps": selected["replan_steps"],
            "development_success_rate": selected["success_rate"],
            "selection_rule": "development success, paired semantic gap, smaller replan",
            "frozen_before_final": True,
        }
        if args.final_only:
            frozen = json.loads(selection_path.read_text())
            if frozen != selection:
                raise RuntimeError("Frozen checkpoint/replan identity differs from development selection")
        else:
            if selection_path.exists():
                raise FileExistsError(selection_path)
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        if args.development_only:
            (dev_path.parent / "stage_status.json").write_text(json.dumps({
                "status": "development_completed_pending_visual_and_training_review",
                "final_test_used": False, "checkpoint_content_sha256": identity["content_tree_sha256"],
                "development_success_rate": selected["success_rate"],
                "selected_replan_steps": selected["replan_steps"],
            }, indent=2) + "\n")
            print("Development completed; final split remains unopened for pipeline/finetune review", flush=True)
            return
        subprocess.run([*common, "--split", "final_test", "--replan-steps", str(selected["replan_steps"])],
                       cwd=ROOT, env=environment, check=True)
        protocol = json.loads((ROOT / "VLA/evaluation_protocol.json").read_text())
        video_test = f"protocol_final_test_{args.run_tag}_seed{protocol['reporting']['combined_video_seed']}_rp{selected['replan_steps']}"
        subprocess.run([str(simulator_python), "-m", "VLA.audit_video", str(ROOT / "VLA" / args.model / video_test)],
                       cwd=ROOT, env=environment, check=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
