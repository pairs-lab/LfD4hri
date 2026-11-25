"""Durable other-VLA queue: native probes, qualified OpenVLA2 ->1k ->120.

Does not interrupt any trainer/evaluation. Unadapted integration probes are
NEVER relabelled as task SR. Every failure and missing action is preserved.
Pi0/FAST/CogACT full fine-tuning is NOT claimed by this queue.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT, gpu_state

SIM = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
OV = ROOT / "VLA/.venvs/openvla/bin/python"
PI = ROOT / "VLA/third_party/openpi/.venv/bin/python"
OV_BASE = str(ROOT / "VLA/checkpoints/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f")


def available_gpu(model, *, allow_rdt_fit=False):
    gpu = gpu_state(); owners = set(map(int, gpu["compute_pids"])); allowed = set()
    if allow_rdt_fit:
        receipt = ROOT / "VLA/audits/rdt_privileged_postfit_r25_20261014/active_child.json"
        if receipt.exists():
            child = json.loads(receipt.read_text()); pid = int(child["pid"])
            proc = Path(f"/proc/{pid}")
            if child["stage"] == "privileged_train" and proc.exists():
                argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                if ("-m VLA.rdt.train_simstate " in argv and
                        "--projection /work/LfD4hri/DRL/VLA/audits/recovery_r17_20261013/train " in argv and
                        "--output /work/LfD4hri/DRL/VLA/rdt_simstate/rdt_privileged_postfit_r25_20261014/train " in argv):
                    allowed.add(pid)
    minimum = 7300 if model == "openvla_integration" and allowed else (14500 if model == "cogact" else 14000)
    return not (owners - allowed) and gpu["free_mib"] >= minimum, gpu, sorted(allowed)


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    parent = output.parent
    receipts = [parent / "language_preflight/audit.json", parent / "openpi_preflight/audit.json"]
    for path in receipts:
        proof = json.loads(path.read_text())
        if not proof.get("passed") or proof.get("neural_weights_loaded") is not False:
            raise ValueError("Actual CPU preflight required before queue")
        for field, source_file in (("prompt_builder_sha256", "privileged_prompt.py"),
                                   ("dataset_implementation_sha256", "privileged_native_dataset.py")):
            if proof[field] != sha256_file(ROOT / "VLA" / source_file): raise ValueError("Audited native input implementation changed")
    files = [Path(__file__), ROOT / "VLA/privileged_native_server.py", ROOT / "VLA/privileged_native_trial.py",
             ROOT / "VLA/privileged_prompt.py", ROOT / "VLA/privileged_native_dataset.py",
             ROOT / "VLA/openvla/train_privileged_prompt.py", ROOT / "VLA/openvla/audit_privileged_saved_precision.py",
             ROOT / "VLA/openvla/adapter.py", ROOT / "VLA/openvla/finetune.py",
             ROOT / "VLA/evaluate_privileged_native120.py", *receipts]
    frozen = {str(p): sha256_file(p) for p in files}
    history = []; models = {}; active = server = None; stage_name = "starting"
    prefix = args.native_prefix.resolve() if args.native_prefix else None
    if prefix is not None:
        previous = json.loads((prefix / "status.json").read_text())
        handoff = json.loads((prefix / "idle_handoff_receipt.json").read_text())
        if (handoff.get("children") != "" or handoff.get("rdt_training_or_robot_episodes_interrupted") is not False
                or previous.get("stage") != "waiting_gpu_for_pi0"):
            raise ValueError("Only an explicitly idle, preserved native-probe prefix can continue")
        for path, digest in previous["frozen_files"].items():
            if Path(path).resolve() != Path(__file__).resolve() and sha256_file(path) != digest:
                raise ValueError("Existing native controls used a different frozen input/runtime: " + path)
        atomic_json(output / "preserved_native_prefix.json", {"source": str(prefix),
            "old_driver_sha256": previous["frozen_files"][str(Path(__file__).resolve())],
            "only_driver_scheduling_and_cache_environment_changed": True,
            "previous_completed_native_neural_controls_not_rerolled": True,
            "prior_analysis_sha256": sha256_file(prefix / "openvla_native8/analysis.json")})
    deadline = time.monotonic() + args.wait_hours * 3600
    def save(status, **extra):
        atomic_json(output / "status.json", {"pid": os.getpid(), "status": status, "stage": stage_name,
            "models": models, "history": history, "frozen_files": frozen,
            "privileged_information_given_to_native_models": True, "oracle_or_ik_action_planner": False,
            "kinematic_motor_servo_retained": True, "all_models_finetuned_or_SR_known": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
    def check():
        for path, digest in frozen.items():
            if sha256_file(path) != digest: raise ValueError("Queued runtime changed; stop before next stage: " + path)
    def stage(name, command, hours=36):
        nonlocal active, stage_name
        stage_name = name; check(); save("running")
        with (output / (name + ".log")).open("x") as log:
            active = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / "active_child.json", {"stage": name, "pid": active.pid, "command": list(map(str, command))})
            until = time.monotonic() + hours * 3600
            while active.poll() is None:
                save("running"); time.sleep(10)
                if time.monotonic() > until: raise TimeoutError("Owned child retained; never kill in-flight neural actions")
        history.append({"stage": name, "exit_code": active.returncode})
        if active.returncode: raise RuntimeError(name + " failed; actual artifacts preserved")
    def wait_gpu(model, allow_fit=False):
        nonlocal stage_name
        stage_name = "waiting_gpu_for_" + model
        while True:
            check(); ready, gpu, allowed = available_gpu(model, allow_rdt_fit=allow_fit)
            save("waiting_gpu_no_training_or_episode_interruption", gpu=gpu, verified_allowed_rdt_fit=allowed)
            if ready: return
            if time.monotonic() > deadline: raise TimeoutError("Finite quiet-GPU deadline reached; no competing trainer killed")
            time.sleep(15)
    def start_server(model, checkpoint, port, name, key="berkeley_autolab_ur5", cap=0):
        nonlocal server, stage_name
        stage_name = "start_" + name; check(); save("running")
        python = PI if model in {"pi0", "pi0_fast"} else OV
        command = [python, "-m", "VLA.privileged_native_server", "--model", model,
            "--checkpoint", checkpoint, "--port", port, "--output", output / (name + "_runtime"), "--unnorm-key", key]
        if cap: command += ["--gpu-memory-cap-mib", str(cap)]
        with (output / (name + "_server.log")).open("x") as log:
            server = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        atomic_json(output / "owned_server.json", {"pid": server.pid, "command": list(map(str, command)), "stage": name})
        import requests
        until = time.monotonic() + 1800
        while True:
            if server.poll() is not None: raise RuntimeError(name + " native server failed to load; not a task failure")
            try:
                response = requests.get(f"http://127.0.0.1:{port}/metadata", timeout=2)
                if response.ok and response.json().get("checkpoint") == str(checkpoint): return
            except requests.RequestException: pass
            save("running"); time.sleep(3)
            if time.monotonic() > until: raise TimeoutError("Native startup timeout; no neural actions underway")
    def close_idle_server():
        nonlocal server
        if server is not None and server.poll() is None:
            if active is not None and active.poll() is None:
                save("owned_inflight_child_and_server_preserved"); return
            server.terminate(); server.wait(timeout=30)
        server = None
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8", "OPENPI_DATA_HOME": str(ROOT / "VLA/checkpoints/openpi"),
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface")}
    environment.pop("JAX_PLATFORMS", None)  # CPU preflight setting must never disable CUDA inference.
    source = Path(json.loads(receipts[0].read_text())["source"])
    specs = [("openvla", OV_BASE, 5741), ("pi0", "gs://openpi-assets/checkpoints/pi0_base", 5742),
        ("pi0_fast", "gs://openpi-assets/checkpoints/pi0_fast_base", 5743),
        ("cogact", str(ROOT / "VLA/checkpoints/cogact-small/checkpoints/CogACT-Small.pt"), 5744)]
    try:
        for model, checkpoint, port in specs:
            models[model] = {"status": "queued", "neural_controls": 0, "task_SR": None,
                "adaptation_regime": "official_weights_privileged_prompt_integration_only"}
        save("running")
        for model, checkpoint, port in specs:
            if prefix is not None and model == "openvla":
                actual = json.loads((prefix / "openvla_native8/analysis.json").read_text())
                served = json.loads((prefix / "openvla_runtime/metadata.json").read_text())
                if (actual.get("status") != "completed_eight_native_neural_controls_NOT_SR"
                        or actual.get("neural_controls_executed") != 8 or served.get("checkpoint") != checkpoint):
                    raise ValueError("Previous native8 is not the same completed official OpenVLA probe")
                models[model].update(status=actual["status"], neural_controls=8, task_SR=None,
                    integration_only_NOT_SR=True, preserved_prefix=str(prefix / "openvla_native8/analysis.json"))
                continue
            wait_gpu("openvla_integration" if model == "openvla" else model, model == "openvla")
            try:
                start_server(model, checkpoint, port, model, cap=6200 if model == "openvla" else 0)
                stage(model + "_native8", [SIM, "-m", "VLA.privileged_native_trial", "--model", model,
                    "--checkpoint", checkpoint, "--port", port, "--source", source, "--output", output / (model + "_native8")])
                actual = json.loads((output / (model + "_native8") / "analysis.json").read_text())
                models[model].update(status=actual["status"], neural_controls=actual["neural_controls_executed"],
                    task_SR=None, integration_only_NOT_SR=True)
            except Exception as error:
                models[model].update(status="native_integration_failed_NOT_scored_task_failure", error=f"{type(error).__name__}: {error}")
                if active is not None and active.poll() is None: raise
            finally:
                close_idle_server(); save("running")
        # Continue only the separately qualified OpenVLA input branch. The
        # other probes do NOT magically authorize their long training stages.
        dataset = args.openvla_dataset.resolve(); contract = dataset / "privileged_dataset.json"
        if not contract.is_file(): raise FileNotFoundError("Audited OpenVLA pilot source preparation not completed")
        parity_path = args.openvla_parity.resolve()
        parity = json.loads(parity_path.read_text())
        if (not parity.get("passed") or parity.get("samples") != 128 or parity.get("actual_loader_batches") != 64
                or parity.get("dataset_contract_sha256") != sha256_file(contract)
                or parity.get("trainer_sha256") != sha256_file(ROOT / "VLA/openvla/train_privileged_prompt.py")
                or any(parity.get("errors", {}).values())):
            raise ValueError("Actual new-input native OpenVLA batching qualification failed")
        frozen[str(parity_path)] = sha256_file(parity_path)
        wait_gpu("openvla_finetune")
        fit = ROOT / "VLA/openvla_privileged_language" / parent.name / "train"
        command = [OV, "-m", "VLA.openvla.train_privileged_prompt", "--dataset", dataset, "--output", fit,
            "--base-model", OV_BASE, "--steps", "2", "--grad-accumulation", "4", "--save-freq", "250"]
        stage("openvla_actual_two_updates", command)
        cp2 = fit / "checkpoints/000002"
        stage("openvla_saved_two_update_precision", [OV, "-m", "VLA.openvla.audit_privileged_saved_precision",
            "--checkpoint", cp2, "--updates", "2", "--output", output / "openvla_saved_two_update_precision.json"])
        start_server("openvla", str(cp2), 5741, "openvla_smoke_reload", key=dataset.name)
        stage("openvla_smoke_native8", [SIM, "-m", "VLA.privileged_native_trial", "--model", "openvla",
            "--checkpoint", cp2, "--port", "5741", "--source", source, "--output", output / "openvla_smoke_native8"])
        close_idle_server()
        precision = json.loads((output / "openvla_saved_two_update_precision.json").read_text())
        reload = output / "openvla_smoke_reload_runtime/adapter_reload.json"
        actual = json.loads((output / "openvla_smoke_native8/analysis.json").read_text())
        if (not precision["passed"] or not json.loads(reload.read_text())["passed"] or actual["neural_controls_executed"] != 8
                or actual["status"] != "completed_eight_native_neural_controls_NOT_SR"):
            raise ValueError("Actual two-update/reload/robot qualification failed")
        paths = [output / "openvla_saved_two_update_precision.json", reload,
                 output / "openvla_smoke_native8/analysis.json", cp2 / "checkpoint_manifest.json"]
        qualification = output / "openvla_actual_smoke_qualification.json"
        atomic_json(qualification, {"passed": True, "optimizer_updates": 2,
            "dataset_contract_sha256": sha256_file(contract),
            "trainer_sha256": sha256_file(ROOT / "VLA/openvla/train_privileged_prompt.py"),
            "grad_accumulation": 4, "learning_rate": .0002, "seed": 20261014,
            "actual_neural_robot_controls": 8, "task_SR_certified": False,
            "dependencies": [{"path": str(p), "sha256": sha256_file(p)} for p in paths]})
        wait_gpu("openvla_finetune")
        command[command.index("--steps") + 1] = "1000"
        stage("openvla_qualified_pilot1000", command + ["--resume-from", cp2, "--smoke-qualification", qualification])
        final = fit / "checkpoints/001000"
        stage("openvla_saved1000_precision", [OV, "-m", "VLA.openvla.audit_privileged_saved_precision",
            "--checkpoint", final, "--updates", "1000", "--output", output / "openvla_saved1000_precision.json"])
        start_server("openvla", str(final), 5741, "openvla_postfit", key=dataset.name)
        stage("openvla_development120", [SIM, "-m", "VLA.evaluate_privileged_native120", "--checkpoint", final,
            "--port", "5741", "--output", output / "openvla_development120"])
        close_idle_server()
        stage_name = "completed_bounded_queue_NOT_all_family_finetuning"; save("completed")
    except Exception as error:
        save("failed_preserve_all_actual_artifacts", error=f"{type(error).__name__}: {error}")
        raise
    finally: close_idle_server()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--openvla-dataset", type=Path, required=True)
    p.add_argument("--openvla-parity", type=Path, required=True)
    p.add_argument("--native-prefix", type=Path)
    p.add_argument("--wait-hours", type=float, default=72); p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    if args.launch:
        if args.output.exists(): raise FileExistsError("Do not duplicate or overwrite a queued cycle")
        log = args.output.parent / (args.output.name + "_launcher.log")
        with log.open("x") as stream:
            command = [sys.executable, "-m", "VLA.run_other_privileged_vlas", "--output", str(args.output.resolve()),
                "--openvla-dataset", str(args.openvla_dataset.resolve()), "--openvla-parity", str(args.openvla_parity.resolve()),
                "--wait-hours", str(args.wait_hours)]
            if args.native_prefix: command += ["--native-prefix", str(args.native_prefix.resolve())]
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.output.parent / (args.output.name + "_launch.json"), {"pid": child.pid, "log": str(log.resolve()),
            "no_existing_trainer_or_evaluation_interrupted": True})
        print(json.dumps({"pid": child.pid, "output": str(args.output.resolve())}), flush=True)
    else: run(args)


if __name__ == "__main__": main()
