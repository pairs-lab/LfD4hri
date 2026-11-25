"""Prioritize qualified repair ahead of an UNSTARTED zero-shot diagnostic queue.

Only exact owned child-free CPU waiters with zero learned episodes may be
superseded. Existing R16 optimization/evaluation owners are NEVER signalled.
All source data/results/plans survive; original120/model is requeued after the
new privileged branch. GPU acquisition waits for BOTH protected cycles.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT, PROTECTED, gpu_state, protected_running
from VLA.run_simstate_repair_cycle import BASE, training_command, finalize_checkpoint

REPAIR = ROOT / "VLA/audits/simstate18_full_repair_r1_20261013"
OUTPUT = ROOT / "VLA/audits/simstate18_priority_r2_20261013"
CROSS = ROOT / "VLA/audits/cross_vla120_per_model_20261013"


def stop_owned_cpu_waiter(status, module, output):
    pid = status["pid"]
    process = Path(f"/proc/{pid}")
    if not process.exists(): raise ValueError("Owned waiter disappeared; audit identity again")
    tokens = (process / "cmdline").read_bytes().split(b"\0")
    expected = [b"-m", module.encode()]
    if not any(tokens[i:i+2] == expected for i in range(len(tokens)-1)) or str(output).encode() not in tokens:
        raise ValueError("PID/module/output identity differs; refuse signalling")
    if (process / "cwd").resolve() != ROOT:
        raise ValueError("Unexpected owned waiter cwd")
    if (process / "task" / str(pid) / "children").read_text().strip():
        raise ValueError("Never stop a waiter with an active collector/train/server child")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while process.exists():
        if time.monotonic() > deadline: raise TimeoutError("Owned waiter did not terminate")
        time.sleep(.2)


def main():
    if OUTPUT.exists(): raise FileExistsError("Immutable priority cycle")
    OUTPUT.mkdir(parents=True)
    history = []
    def save(stage, state="running", error=None):
        atomic_json(OUTPUT / "status.json", {"pid": os.getpid(), "status": state, "stage": stage,
            "error": error, "history": history, "CPU_projections_preserved": str(REPAIR),
            "privileged_VLA_branch_not_plain_VLA": True, "original120_per_model_preserved": True,
            "no_existing_trainer_signalled": True, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    def run(stage, command, extra=None):
        save(stage)
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB": "10000",
            "VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR": "1", "VLA_SMOLVLA_INITIAL_FRAME_PROB": ".1",
            "VLA_SMOLVLA_TARGET_BALANCED": "1", "VLA_SMOLVLA_GROUNDING_PROB": "0",
            "VLA_SMOLVLA_SAMPLING_SOURCE": str(ROOT / "VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train"),
            **(extra or {})}
        with (OUTPUT / f"{stage}.log").open("x") as log:
            done = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=86400)
        history.append({"stage": stage, "exit_code": done.returncode})
        if done.returncode: raise RuntimeError(f"{stage} failed; preserve evidence, no silent retry")
    deadline = time.monotonic() + 72*3600
    try:
        while True:
            repair = json.loads((REPAIR / "status.json").read_text())
            if repair["status"] == "failed": raise RuntimeError("CPU repair audit failed; no unsupported training")
            if repair["stage"] == "qualified_CPU_data_waiting_existing_cycles_and_GPU": break
            if not Path(f"/proc/{repair['pid']}").exists(): raise RuntimeError("CPU repair owner exited before qualification")
            save("waiting_full_TRAIN_VAL_and_actual_loader_audit")
            if time.monotonic() > deadline: raise TimeoutError("Finite CPU/GPU qualification wait expired")
            time.sleep(15)
        cross = json.loads((CROSS / "progress.json").read_text())
        if cross["status"] != "waiting_gpu" or cross["completed_neural_trials"] != 0:
            raise ValueError("Baseline testing already started; do not discard or interrupt it")
        stop_owned_cpu_waiter(repair, "VLA.run_simstate_repair_cycle", REPAIR)
        atomic_json(REPAIR / "status.json", {**repair, "status": "superseded_owned_CPU_waiter_priority_r2",
            "stage": "CPU_qualification_preserved_NO_optimization_started", "priority_cycle": str(OUTPUT)})
        stop_owned_cpu_waiter(cross, "VLA.run_cross_vla120", CROSS)
        atomic_json(CROSS / "progress.json", {**cross, "status": "deferred_for_user_requested_repair_before_zero_shot",
            "error": None, "priority_cycle": str(OUTPUT), "episodes_discarded": 0, "checkpoint_seed_rules_changed": False})
        while True:
            protected = protected_running({"protected_status_files": list(map(str, PROTECTED))})
            gpu = gpu_state()
            if not protected and not gpu["compute_pids"] and gpu["free_mib"] >= 11000: break
            save("qualified_inputs_waiting_PROTECTED_R16_cycles_and_GPU")
            if time.monotonic() > deadline: raise TimeoutError("Finite GPU wait expired")
            time.sleep(30)
        from VLA.smolvla.simstate_dataset import load_projection
        from VLA.fingerprint_checkpoint import verify_fingerprint
        load_projection(REPAIR/"train")
        verify_fingerprint(BASE, full=True)
        projection = REPAIR / "train"
        model_root = ROOT / "VLA/smolvla_simstate" / OUTPUT.name / "test1"
        for label, updates, accumulation in [("smoke",2,2),("train",5000,8)]:
            fit = model_root/label
            run(label, training_command(BASE, projection, fit, updates, accumulation), {
                "VLA_SIMSTATE_TRAIN_PROJECTION": str(projection), "VLA_SIMSTATE_OPTIMIZER_UPDATES": str(updates),
                "VLA_SMOLVLA_MASTER_AUDIT": str(fit / "master_dtype_audit.json")})
            cp = finalize_checkpoint(fit, updates, accumulation, projection)
            run(label+"_precision", [sys.executable,"-m","VLA.smolvla.audit_training_precision",
                "--checkpoint",cp,"--output",OUTPUT/(label+"_precision.json")])
            run(label+"_fingerprint", [sys.executable,"-m","VLA.fingerprint_checkpoint",cp])
            if label == "smoke":
                run("new_input_weight_updates", [sys.executable,"-m","VLA.smolvla.audit_simstate_updates",
                    "--base",BASE,"--checkpoint",cp,"--output",OUTPUT/"new_input_weight_updates.json"])
        run("learned120_simstate", [sys.executable,"-m","VLA.evaluate_simstate120",
            "--checkpoint",cp,"--output",OUTPUT/"development120"])
        # Resume the original declared120/model with a NEW immutable directory,
        # SAME seeds/checkpoints/scoring, not adopt or overwrite old artifacts.
        resumed = ROOT / "VLA/audits/cross_vla120_after_simstate_20261013"
        run("original120_each_model", [sys.executable,"-m","VLA.run_cross_vla120",
            "--initialize","--output",resumed,"--wait-hours","48"])
        save("privileged120_and_original_model_cohort_finished_requires_actual_success_review", state="completed")
    except Exception as error:
        save("stopped_with_preserved_evidence_no_false_success", state="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
