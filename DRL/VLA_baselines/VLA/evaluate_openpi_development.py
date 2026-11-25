"""Source-explicit OpenPI development evaluation; never opens final-test seeds.

Optionally wait for the identified trainer to exit, then require its finalized
checkpoint. No automatic resume, threshold changes or oracle policy fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        # comm may contain spaces/parentheses. Fields after its final ')' start
        # with state (field 3); starttime is field 22, index 19 in that suffix.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def validate_checkpoint_contract(checkpoint, model, repo_id, source):
    contract = json.loads((checkpoint / "vla_checkpoint_manifest.json").read_text())
    if contract["model"] != model or contract["repo_id"] != repo_id:
        raise RuntimeError("Checkpoint model/dataset differs from the requested evaluation")
    if Path(contract["source"]["source"]).resolve() != source.resolve():
        raise RuntimeError("Checkpoint belongs to another source dataset")
    if contract["optimization"]["partial_accumulation_microsteps"] != 0:
        raise RuntimeError("Evaluate only completed optimizer-update boundaries")
    return contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi0_fast"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    for name in ("audit-report", "roundtrip-report", "action-adapter-report", "closed-loop-adapter-report", "oracle-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "VLA/evaluation_protocol_v8.json")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--wait-for-training-pid", type=int)
    parser.add_argument("--wait-timeout-hours", type=float, default=24)
    args = parser.parse_args()
    from VLA.run_evaluation_protocol import validate_protocol
    protocol, _ = validate_protocol(args.protocol, args.oracle_audit, requested_split="development")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        raise ValueError("run-tag must be one safe path component")
    if args.replan_steps not in protocol["action_chunk_selection"]["development_candidates"]:
        raise ValueError("Replan setting is not in the locked development protocol")
    output = ROOT / "VLA/evaluations" / args.model / "development" / args.run_tag
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "driver_status.json"
    if status_path.exists() or (output / "aggregate.json").exists():
        raise FileExistsError("Evaluation label already exists")
    history = []

    def status(state, stage):
        status_path.write_text(json.dumps({"status": state, "stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(), "history": history,
            "checkpoint": str(args.checkpoint.resolve()), "final_test_used": False}, indent=2) + "\n")

    env = os.environ.copy()
    env.pop("JAX_PLATFORMS", None)
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
        "OPENPI_DATA_HOME": str(ROOT / "VLA/cache/openpi"), "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80"})
    server = stream = None
    try:
        if args.wait_for_training_pid:
            identity = process_identity(args.wait_for_training_pid)
            deadline = time.monotonic() + args.wait_timeout_hours * 3600
            while identity is not None and process_identity(args.wait_for_training_pid) == identity:
                status("waiting_for_trainer", "checkpoint")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Training wait exceeded bound; no checkpoint evaluation attempted")
                time.sleep(15)
        contract = validate_checkpoint_contract(args.checkpoint, args.model, args.repo_id, args.source_dataset)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError(f"GPU still has a compute job ({active}); do not launch a competing inference model")
        from VLA.fingerprint_checkpoint import verify_fingerprint
        if not (args.checkpoint / "benchmark_fingerprint.json").exists():
            subprocess.run([sys.executable, "-m", "VLA.fingerprint_checkpoint", str(args.checkpoint)], cwd=ROOT, env=env, check=True)
        verify_fingerprint(args.checkpoint)
        try:
            with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                raise RuntimeError("Evaluation server port is already occupied")
        except (ConnectionRefusedError, TimeoutError):
            pass
        command = [str(ROOT / "VLA/third_party/openpi/.venv/bin/python"), "-m", "VLA.pi0.server",
            "--model", args.model, "--checkpoint", str(args.checkpoint), "--repo-id", args.repo_id,
            "--source-dataset", str(args.source_dataset), "--action-horizon", str(contract["action_horizon"]),
            "--port", str(args.port)]
        for name in ("audit_report", "roundtrip_report", "action_adapter_report", "closed_loop_adapter_report"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        log = ROOT / "VLA/logs" / (args.run_tag + "_development_server.log")
        stream = log.open("x")
        status("starting_server", "server")
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Inference server failed; inspect {log}")
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server did not become ready; inspect {log}")
                time.sleep(2)
        commands = [
            ("development", ["-m", "VLA.run_evaluation_protocol", "--model", args.model,
                "--split", "development", "--checkpoint", str(args.checkpoint), "--run-tag", args.run_tag,
                "--protocol", str(args.protocol), "--oracle-audit", str(args.oracle_audit),
                "--replan-steps", str(args.replan_steps), "--port", str(args.port)]),
            ("video_repeat", ["-m", "VLA.benchmark", "--model", args.model, "--checkpoint", str(args.checkpoint),
                "--port", str(args.port), "--scene-profile", "rl_random_env", "--seed", "20262001",
                "--replan-steps", str(args.replan_steps), "--required-success-steps", "1",
                "--success-mode", "semantic", "--audit-observation-every", "10",
                "--test-id", args.run_tag + "_dev_seed20262001_video_repeat"]),
        ]
        for stage, command in commands:
            status("running", stage)
            log = ROOT / "VLA/logs" / (args.run_tag + "_" + stage + ".log")
            with log.open("x") as handle:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            history.append({"stage": stage, "exit_code": result.returncode, "log": str(log)})
            if result.returncode:
                raise RuntimeError(f"Evaluation failed at {stage}; inspect {log}")
        status("development_completed_pending_visual_review", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", "integration")
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
