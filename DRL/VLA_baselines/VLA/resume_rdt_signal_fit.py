"""Preserve stopped R14; prove actual signal integration -> fresh fit ->40.

Completed CPU/source/smoke proofs are reused only if byte-bound and successful.
Owns every server. No failed trial filtering, scoring changes or fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import requests

from VLA.common.provenance import sha256_file
from VLA.rdt.signal_qualification import build_signal_qualification

ROOT = Path(__file__).resolve().parents[1]
MODEL_PYTHON = ROOT / "VLA/.venvs/rdt/bin/python"
SIM_PYTHON = ROOT / "VLA/third_party/lerobot/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stopped-audit", "output", "model-output", "wait-report"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=5628)
    parser.add_argument("--reuse-smoke-integration", type=Path,
                        help="Preserve and reuse already executed complete four-task smoke, never reroll")
    parser.add_argument("--guard-replay", type=Path, help="Independent exact command/rollback proof if smoke safety-aborted")
    args = parser.parse_args()
    old, output, model = args.stopped_audit.resolve(), args.output.resolve(), args.model_output.resolve()
    state = json.loads((old / "status.json").read_text())
    if state["status"] != "stopped_error" or state["stage"] != "train" or "train failed" not in state["error"]:
        raise ValueError("Only the diagnosed old long-fit guard stop is covered")
    if "only a separate <=2-update smoke" not in (old / "train.log").read_text():
        raise ValueError("Failure is not the diagnosed missing signal qualification")
    if output.exists() or model.exists():
        raise FileExistsError("Fresh continuation/model paths required")
    steps = {row["name"]: row for row in state["stages"]}
    required = {"source_audit", "native_fk", "language", "tensor_parity", "input_smoke",
        "train_deployment_reset_parity", "heldout_deployment_reset_parity", "gpu_headroom",
        "smoke", "smoke_master_adam", "smoke_reload_precision", "smoke_fingerprint"}
    if not required <= {row["stage"] for row in state["history"] if row["exit_code"] == 0}:
        raise ValueError("Missing successful completed source/smoke stages")
    source = Path(state["source"])
    source_audit = old / "qualified_source_audit/audit.json"
    smoke = Path(steps["smoke"]["command"][steps["smoke"]["command"].index("--output")+1]) / "checkpoints/step-000002"
    from VLA.common.source_training_gate import require_training_source
    require_training_source(source, model="rdt", optimizer_updates=state["updates"])
    fixed = {str(p): sha256_file(p) for p in (old / "status.json", old / "train.log", source / "index.json", source / "dataset_manifest.json", Path(__file__),
        ROOT / "VLA/rdt/training.py", ROOT / "VLA/rdt/signal_qualification.py", ROOT / "VLA/rdt/server.py",
        ROOT / "VLA/evaluate_rdt_v9_source.py", ROOT / "VLA/run_rdt_diagnostic40.py")}
    for name in required:
        report = steps[name].get("report")
        if report:
            p = Path(report)
            data = json.loads(p.read_text())
            if data.get("passed") is not True or data.get("failures"):
                raise ValueError("Failed completed gate: " + name)
            fixed[str(p)] = sha256_file(p)
    output.mkdir(parents=True)
    model.mkdir(parents=True)
    history = []
    status = {"scope": "diagnosed_RDT_signal_guard_fix_fresh5k_then40_DEVELOPMENT", "old_failed_run_preserved": str(old),
        "source": str(source), "model_output": str(model), "updates": state["updates"], "pid": os.getpid(),
        "frozen_files": fixed, "history": history, "status": "running", "error": None,
        "final_test_used": False, "oracle_fallback": False, "high_success_guaranteed": False}
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "TRANSFORMERS_OFFLINE": "1"}
    def save(stage):
        status.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat())
        (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in fixed.items()):
            raise ValueError("Bound continuation/source implementation changed")
    def run(name, command):
        verify(); save(name)
        with (output / (name+".log")).open("x") as log:
            done = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=86400)
        history.append({"stage": name, "command": list(map(str,command)), "exit_code": done.returncode})
        if done.returncode: raise RuntimeError(name+" failed; diagnose preserved log")
        save(name+"_completed")
    server, stream = None, None
    client = requests.Session(); client.trust_env = False
    def stop_server():
        nonlocal server, stream
        if server is not None and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=15)
        if stream is not None: stream.close()
        server, stream = None, None
    def start_server(cp, label):
        nonlocal server, stream
        with socket.socket() as sock: sock.bind(("127.0.0.1", args.port))
        stream = (output / (label+"_server.log")).open("x")
        server = subprocess.Popen([str(MODEL_PYTHON), "-m", "VLA.rdt.server", "--checkpoint", str(cp),
            "--embeddings", str(old / "qualified_embeddings.pt"), "--vision", str(ROOT / "VLA/checkpoints/siglip-so400m-patch14-384"),
            "--port", str(args.port)], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+900
        while True:
            if server.poll() is not None or time.monotonic() > deadline: raise RuntimeError("Owned server failed/timed out")
            try:
                response = client.get(f"http://127.0.0.1:{args.port}/metadata", timeout=2); response.raise_for_status()
                data = response.json()
                if Path(data["checkpoint"]["path"]).resolve() != cp.resolve(): raise ValueError("Wrong served checkpoint")
                return data
            except requests.RequestException: time.sleep(2)
    def source_rollout(cp, dest, cap=None):
        command = [SIM_PYTHON, "-m", "VLA.evaluate_rdt_v9_source", "--checkpoint", cp, "--source", source,
            "--audit-report", source_audit, "--training-source", source, "--training-audit", source_audit,
            "--split-mode", "train_diagnostic", "--port", args.port, "--episodes-per-task", "1", "--output", dest]
        if cap is not None: command += ["--integration-step-cap", cap]
        return command
    try:
        deadline = time.monotonic()+7200
        while not args.wait_report.exists():
            verify(); save("waiting_owned_SmolVLA_grounding_GPU_audit")
            if time.monotonic() > deadline: raise TimeoutError("Previous GPU audit did not finish")
            time.sleep(20)
        previous = json.loads(args.wait_report.read_text())
        if not previous.get("sensor_identity_passed") or not previous.get("full_text_not_truncated"):
            raise ValueError("Previous grounding audit failed integration")
        run("gpu_headroom", steps["gpu_headroom"]["command"])
        if args.reuse_smoke_integration:
            integration = args.reuse_smoke_integration.resolve()
            save("reuse_complete_smoke_preserve_original_guard_event")
        else:
            save("owned_smoke_deployment_integration")
            metadata = start_server(smoke, "smoke")
            integration = output / "smoke_integration"
            run("smoke_four_task_integration", source_rollout(smoke, integration, 2))
            (integration / "server_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
            stop_server()
        proof = output / "signal_qualification.json"
        build_signal_qualification(source, smoke, old, integration / "results.json", proof, args.guard_replay)
        command = steps["train"]["command"].copy()
        command[command.index("--output")+1] = str(model / "train")
        command += ["--signal-qualification", str(proof)]
        run("fresh_train", command)
        cp = model / "train/checkpoints" / f"step-{state['updates']:06d}"
        for name in ("train_master_adam", "train_reload_precision", "train_fingerprint"):
            command = steps[name]["command"].copy()
            if "--checkpoint" in command: command[command.index("--checkpoint")+1] = str(cp)
            else: command[-1] = str(cp)
            if "--output" in command: command[command.index("--output")+1] = str(output / (name+".json"))
            run(name, command)
            if "--output" in command and json.loads(Path(command[command.index("--output")+1]).read_text()).get("passed") is not True:
                raise ValueError(name+" report failed")
        save("owned_postfit_server")
        start_server(cp, "postfit")
        run("source_diagnostic", source_rollout(cp, model / "train_closedloop"))
        run("development40", [SIM_PYTHON, "-m", "VLA.run_rdt_diagnostic40", "--checkpoint", cp, "--output", output / "development40",
            "--port", args.port, "--run-tag", model.parent.name+"_postfit40"])
        status["status"] = "completed_fit_test_remaining_errors_require_review"; save("done")
    except Exception as exc:
        status.update(status="stopped_error", error=str(exc)); save(status.get("stage", "initialization")); raise
    finally:
        stop_server()


if __name__ == "__main__":
    main()
