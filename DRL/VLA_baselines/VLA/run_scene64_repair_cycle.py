"""Finite, auditable data repair alongside protected existing VLA jobs.

Fresh full-source physical replay (CPU) and40 real student TRAIN prefixes
(bounded GPU) run independently. Replay follows collection automatically.
Neither candidate is silently injected into a live fit. All existing fits,
native probes and benchmark episodes are left intact.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT, gpu_state

SIM = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
BASE = ROOT / "VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013"
CHECKPOINT = ROOT / "VLA/smolvla_simstate/privileged_batchfix_r22_20261014/simstate_train/checkpoints/040000/pretrained_model"


def allowed_collection_gpu():
    state = gpu_state(); allowed = set()
    receipt = ROOT / "VLA/audits/rdt_privileged_postfit_r25_20261014/active_child.json"
    if receipt.exists():
        child = json.loads(receipt.read_text()); pid = int(child["pid"])
        path = Path(f"/proc/{pid}/cmdline")
        if child["stage"] == "privileged_train" and path.exists():
            argv = path.read_bytes().replace(b"\0", b" ").decode()
            if ("-m VLA.rdt.train_simstate " in argv and
                "--output /work/LfD4hri/DRL/VLA/rdt_simstate/rdt_privileged_postfit_r25_20261014/train " in argv):
                allowed.add(pid)
    # Unknown evaluators/trainers are never allowed. This leaves at least
    # 1500MiB headroom beyond the student's enforced3000MiB maximum.
    return not (set(map(int, state["compute_pids"])) - allowed) and state["free_mib"] >= 4500, state, sorted(allowed)


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    root = output.parent
    files = [Path(__file__), ROOT / "VLA/project_scene64.py", ROOT / "VLA/simstate_scene64.py",
        ROOT / "VLA/collect_scene64_recovery40.py", ROOT / "VLA/replay_scene64_recovery.py",
        ROOT / "VLA/staged_acquisition_teacher.py", ROOT / "VLA/collect_demos.py",
        root / "pilot8/audit.json", BASE / "qualified_train/index.json", BASE / "validation_candidate/index.json"]
    frozen = {str(p): sha256_file(p) for p in files}
    pilot = json.loads((root / "pilot8/audit.json").read_text())
    if not pilot["passed"] or pilot["declared_episodes"] != 8 or pilot["frames"] != 407:
        raise ValueError("Actual pilot8/407 evidence required")
    children = {}; history = []; stage = "starting"
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    def check():
        for path, digest in frozen.items():
            if sha256_file(path) != digest: raise ValueError("Frozen repair source changed: " + path)
    def save(status, **extra):
        atomic_json(output / "status.json", {"pid": os.getpid(), "status": status, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "children": {
                name: {"pid": child.pid, "running": child.poll() is None, "exit_code": child.poll()}
                for name, child in children.items()}, "history": history, "frozen_files": frozen,
            "training_qualified": False, "model_SR_improvement_established": False,
            "deploy_actions_generated_by_VLA": True, "expert_only_produces_offline_training_labels": True,
            "existing_model_jobs_interrupted": False, **extra})
    def start(name, command):
        check()
        with (output / (name + ".log")).open("x") as log:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children[name] = child; atomic_json(output / (name + "_child.json"), {
            "pid": child.pid, "command": list(map(str, command))})
        return child
    try:
        full = start("full_source64", [SIM, "-m", "VLA.project_scene64", "--source", BASE / "qualified_train",
            "--source-audit", BASE / "qualified_train_audit/audit.json", "--previous-projection",
            ROOT / "VLA/audits/recovery_r17_20261013/train", "--output", root / "full_train64",
            "--pilot-receipt", root / "pilot8/audit.json"])
        stage = "waiting_bounded_gpu_for_actual_student40"; deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            check(); ready, gpu, owners = allowed_collection_gpu()
            save("running_CPU_replay_waiting_safe_GPU", gpu=gpu, allowed_live_rdt_fit_pids=owners)
            if ready: break
            if time.monotonic() > deadline: raise TimeoutError("Finite GPU wait; no protected job killed")
            time.sleep(15)
        collector = start("recovery40", [SIM, "-m", "VLA.collect_scene64_recovery40",
            "--source", BASE / "qualified_train", "--source-audit", BASE / "qualified_train_audit/audit.json",
            "--validation", BASE / "validation_candidate", "--validation-audit", BASE / "validation_audit/audit.json",
            "--checkpoint", CHECKPOINT, "--output", root / "recovery40"])
        stage = "full64_and_actual_neural_prefix40"; save("running")
        while collector.poll() is None:
            check(); save("running"); time.sleep(15)
            if time.monotonic() > deadline: raise TimeoutError("Owned actions retained, no mid-episode interruption")
        history.append({"stage": "recovery40", "exit_code": collector.returncode})
        if collector.returncode == 0:
            replay = start("recovery_replay", [SIM, "-m", "VLA.replay_scene64_recovery",
                "--candidate", root / "recovery40", "--output", root / "recovery40_replay"])
            stage = "automatic_all_attempts_canonical_native_replay"
            while replay.poll() is None:
                check(); save("running"); time.sleep(15)
            history.append({"stage": "recovery_replay", "exit_code": replay.returncode})
        stage = "waiting_full_source64_extraction"
        while full.poll() is None:
            check(); save("running"); time.sleep(15)
            if time.monotonic() > deadline: raise TimeoutError("Full-source audit retained, finite deadline")
        history.append({"stage": "full_source64", "exit_code": full.returncode})
        check(); stage = "data_audits_finished_next_native64_model_qualification_required"
        save("completed_data_cycle_NOT_finetuned_or_SR_certificate", remaining=[
            "Audit every actual student/corrective failure and raw model-input camera views",
            "Explicit64 input config (old32 padding must not truncate new feedback)",
            "TRAIN-only sampling/normalization and native batch parity",
            "Actual CUDA updates + saved precision + exact reload + robot smoke gate",
            "Family-owned fine-tuning/evaluation behind current protected GPU jobs"])
    except Exception as error:
        save("failed_preserve_children_and_artifacts", error=f"{type(error).__name__}: {error}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--wait-hours", type=float, default=72)
    p.add_argument("--launch", action="store_true"); args = p.parse_args()
    if args.launch:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        marker = args.output.with_name(args.output.name + "_launch.json")
        if marker.exists() or args.output.exists(): raise FileExistsError("Never reroll a registered repair cycle")
        with args.output.with_name(args.output.name + "_launcher.log").open("x") as log:
            child = subprocess.Popen([str(SIM), "-m", "VLA.run_scene64_repair_cycle", "--output", str(args.output.resolve()),
                "--wait-hours", str(args.wait_hours)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {"pid": child.pid, "output": str(args.output.resolve()), "no_existing_jobs_killed": True})
        print(json.dumps({"pid": child.pid, "output": str(args.output.resolve())}))
    else: run(args)


if __name__ == "__main__": main()
