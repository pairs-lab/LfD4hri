"""Finite RDT-specific audit -> 2-update smoke -> fresh5k -> four-task videos.

No fallback, resume from old weights, source mutation, final-test evaluation or
success-threshold change. Stops on the FIRST failed gate; owns its own server.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "validation", "validation-audit", "cache", "audit-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--port", type=int, default=5621)
    parser.add_argument("--signal-precision", choices=("bf16", "fp32_state_action_readout_latent_v1"), default="bf16")
    parser.add_argument("--wait-for-cycle-status", type=Path,
                        help="Prepare on CPU now, queue GPU work until the declared SmolVLA40 fit/test job completes")
    parser.add_argument("--development40", action="store_true", help="After source diagnostics, run the same fixed10/action40 learned trials")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if type(args.updates) is not int or not 1 <= args.updates <= 5000:
        raise ValueError("RDT refinement is bounded to at most5k real updates")
    for name in ("source", "validation", "validation_audit", "cache", "audit_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def build_stages(args):
    model_python = ROOT / "VLA/.venvs/rdt/bin/python"
    sim_python = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
    audit = args.audit_root
    source_audit = audit / "qualified_source_audit/audit.json"
    fk, embeddings = audit / "qualified_fk.json", audit / "qualified_embeddings.pt"
    vision = ROOT / "VLA/checkpoints/siglip-so400m-patch14-384"
    stages = []
    def stage(name, module, tokens, python=model_python, gpu=False, report=None):
        stages.append({"name": name, "command": [str(python), "-m", module, *map(str, tokens)],
                       "gpu": gpu, "report": str(report) if report else None})
    index = args.source / "index.json"
    counts = Counter(row["task"]["name"] for row in json.loads(index.read_text())) if index.exists() else None
    if counts is not None and set(counts) != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("RDT source must retain all four declared tasks")
    count_tokens = ["--expected-task-counts", json.dumps(counts)] if counts is not None else []
    stage("source_audit", "VLA.audit_finetune_data", ["--dataset", args.source, "--output", source_audit.parent,
          *count_tokens, "--allow-stricter-frozen-reward-contract", "--recovery-reset-observability", "dual_view"], python=sim_python, report=source_audit)
    stage("native_fk", "VLA.rdt.audit_action_contract", ["--source", args.source, "--output", fk], report=fk)
    stage("language", "VLA.rdt.rebind_declared_prompt_cache", ["--cache", args.cache, "--new-index", args.source / "index.json", "--output", embeddings])
    parity = audit / "qualified_tensor_parity.json"
    stage("tensor_parity", "VLA.rdt.audit_train_infer_inputs", ["--source", args.source, "--audit-report", source_audit,
          "--action-contract-report", fk, "--embeddings", embeddings, "--vision", vision, "--output", parity], report=parity)
    common = ["--source", args.source, "--audit-report", source_audit, "--action-contract-report", fk,
              "--language-embeddings", embeddings, "--base-checkpoint", ROOT / "VLA/checkpoints/rdt-170m",
              "--vision-checkpoint", vision, "--action-horizon", "10", "--samples-per-episode", "4",
              "--episode-batch-size", "1", "--gradient-accumulation", "4", "--optimizer", "adamw",
              "--num-workers", "1", "--learning-rate", "0.0001", "--seed", "20261010"]
    common += ["--signal-precision", args.signal_precision]
    input_report = audit / "input_smoke.json"
    stage("input_smoke", "VLA.rdt.training", ["smoke", *common, "--steps", "2", "--smoke-report", input_report], report=input_report)
    for split, source, proof in (("train", args.source, source_audit), ("heldout", args.validation, args.validation_audit)):
        report = audit / (split + "_deployment_reset_parity.json")
        stage(split + "_deployment_reset_parity", "VLA.audit_source_reset_observations",
              ["--source", source, "--audit-report", proof, "--episodes-per-task", "2", "--output", report], python=sim_python, report=report)
    stages.append({"name": "gpu_headroom", "gpu": True, "report": None,
        "command": [str(model_python), "-c", "import torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); assert free >= 12*2**30, f'GPU free headroom insufficient: {free}'; print(torch.cuda.get_device_name(0),free,total)"]})
    for name, updates in (("smoke", 2), ("train", args.updates)):
        output = args.output / name
        cp = output / "checkpoints" / f"step-{updates:06d}"
        stage(name, "VLA.rdt.training", ["train", *common, "--steps", updates, "--save-interval", 2 if name == "smoke" else 1000,
               "--output", output], gpu=True)
        report = audit / (name + "_master_adam.json")
        stage(name + "_master_adam", "VLA.rdt.audit_training_precision", ["--checkpoint", cp, "--updates", updates, "--output", report], report=report)
        report = audit / (name + "_reload_precision.json")
        stage(name + "_reload_precision", "VLA.rdt.audit_deployment_precision", ["--checkpoint", cp, "--output", report], report=report)
        stage(name + "_fingerprint", "VLA.fingerprint_checkpoint", [cp])
    return stages


def main():
    args = parse_args()
    stages = build_stages(args)
    if not args.run:
        print(json.dumps({"scope": "bounded_rdt_refinement_not_final_benchmark", "stages": stages}, indent=2))
        return
    from VLA.common.source_training_gate import require_training_source
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
    require_training_source(args.source, model="rdt", optimizer_updates=args.updates)
    validate_holdout(args.source, args.validation)
    if args.output.exists() or args.audit_root.exists():
        raise FileExistsError("Use fresh immutable run and audit directories")
    args.output.mkdir(parents=True)
    args.audit_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                       HF_HOME=str(ROOT / "VLA/checkpoints/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    status = {"scope": "bounded_rdt_refinement_not_final_benchmark", "source": str(args.source),
              "source_index_sha256": sha256_file(args.source / "index.json"), "updates": args.updates,
              "driver_sha256": sha256_file(Path(__file__)), "stages": stages, "history": [],
              "validation_used_for_training": False, "final_test_used": False, "status": "running", "error": None}
    state_path = args.audit_root / "status.json"
    def save(stage):
        status.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(status, indent=2) + "\n")
    def execute(stage):
        if stage["name"] == "gpu_headroom" and args.wait_for_cycle_status:
            deadline = time.monotonic() + 48 * 3600
            while True:
                waited = json.loads(args.wait_for_cycle_status.read_text())
                if waited["status"] == "completed40_fit_test_cycle_remaining_errors_require_review":
                    break
                if waited["status"].startswith("stopped") or time.monotonic() >= deadline:
                    raise RuntimeError("Declared previous GPU cycle failed or finite queue wait expired")
                save("waiting_previous_SmolVLA40_fit_test_GPU_cycle")
                time.sleep(20)
        save(stage["name"])
        log_path = args.audit_root / (stage["name"] + ".log")
        with log_path.open("w") as stream:
            result = subprocess.run(stage["command"], cwd=ROOT, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=86400 if stage["name"] == "train" else 3600)
        status["history"].append({"stage": stage["name"], "exit_code": result.returncode, "log": str(log_path)})
        if result.returncode:
            raise RuntimeError(f"{stage['name']} failed; preserved log: {log_path}")
        if stage.get("report"):
            report = json.loads(Path(stage["report"]).read_text())
            if report.get("passed") is not True or report.get("failures"):
                raise RuntimeError(f"{stage['name']} report failed")
        save(stage["name"] + "_passed")
    server = None
    server_log = None
    try:
        for stage in stages:
            execute(stage)
        with socket.socket() as port_check:
            port_check.bind(("127.0.0.1", args.port))
        cp = args.output / "train/checkpoints" / f"step-{args.updates:06d}"
        server_log = (args.audit_root / "server.log").open("w")
        command = [str(ROOT / "VLA/.venvs/rdt/bin/python"), "-m", "VLA.rdt.server", "--checkpoint", str(cp),
                   "--embeddings", str(args.audit_root / "qualified_embeddings.pt"), "--vision",
                   str(ROOT / "VLA/checkpoints/siglip-so400m-patch14-384"), "--port", str(args.port)]
        save("server_start")
        server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
        import requests
        client = requests.Session()
        client.trust_env = False
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("RDT server failed/timed out; see server.log")
            try:
                response = client.get(f"http://127.0.0.1:{args.port}/metadata", timeout=2)
                response.raise_for_status()
                metadata = response.json()
                if metadata.get("model") != "rdt" or Path(metadata["checkpoint"]["path"]).resolve() != cp.resolve():
                    raise RuntimeError("Dedicated RDT server identity mismatch")
                break
            except requests.RequestException:
                time.sleep(2)
        for split, source, audit in (("heldout", args.validation, args.validation_audit),
                                     ("train", args.source, args.audit_root / "qualified_source_audit/audit.json")):
            command = [str(ROOT / "VLA/third_party/lerobot/.venv/bin/python"), "-m", "VLA.evaluate_rdt_v9_source",
                "--checkpoint", str(cp), "--source", str(source), "--audit-report", str(audit),
                "--training-source", str(args.source), "--training-audit", str(args.audit_root / "qualified_source_audit/audit.json"),
                "--split-mode", "heldout" if split == "heldout" else "train_diagnostic", "--port", str(args.port),
                "--episodes-per-task", "2", "--output", str(args.output / (split + "_closedloop"))]
            execute({"name": split + "_closedloop", "command": command, "gpu": True, "report": None})
        if args.development40:
            execute({"name": "development40", "command": [
                str(ROOT / "VLA/third_party/lerobot/.venv/bin/python"), "-m", "VLA.run_rdt_diagnostic40",
                "--checkpoint", str(cp), "--output", str(args.audit_root / "development40"),
                "--port", str(args.port), "--run-tag", args.output.parent.name + "_postfit40"],
                "gpu": True, "report": None})
        status["status"] = "completed_diagnostic_only"
        save("done")
    except Exception as exc:
        status.update(status="stopped_error", error=str(exc))
        save(status.get("stage", "initialization"))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
