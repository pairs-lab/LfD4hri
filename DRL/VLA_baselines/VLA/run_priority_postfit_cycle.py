"""Test completed Smol/RDT first, then fit/test the separate privileged VLA.

Adopts ONLY a positively identified CPU validation replay and finalized model
artifacts. No trainer/evaluator is interrupted, no model/physics changes and
no outcome-based checkpoint/seed selection. Other four base diagnostics follow.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA import run_cross_vla120 as queue
from VLA.recover_interrupted_cycles import ROOT, SIM, SOURCE, SMOL_FINAL, RDT_FINAL, pid_alive

PREVIOUS = ROOT / "VLA/audits/recovery_r19_metadata_gate_20261014"
TRAIN_PROJECTION = ROOT / "VLA/audits/recovery_r17_20261013/train"


def validate_handoff(previous=PREVIOUS):
    state = json.loads((previous / "status.json").read_text())
    if state["status"] != "running" or state["stage"] != "project_validation":
        raise ValueError("Only the known CPU validation wait can be handed off")
    pid = state["pid"]
    proc = Path(f"/proc/{pid}")
    tokens = (proc / "cmdline").read_bytes().split(b"\0")
    if (not pid_alive(pid) or b"VLA.recover_interrupted_cycles" not in tokens
            or str(previous.resolve()).encode() not in tokens or (proc / "cwd").resolve() != ROOT):
        raise ValueError("Previous passive coordinator identity differs")
    children = (proc / "task" / str(pid) / "children").read_text().split()
    if len(children) != 1:
        raise ValueError("Unexpected children; no process is signalled")
    child = int(children[0])
    child_proc = Path(f"/proc/{child}")
    command = (child_proc / "cmdline").read_bytes().split(b"\0")
    validation = previous / "validation"
    if (not pid_alive(child) or b"VLA.resume_simstate_projection" not in command
            or str(validation.resolve()).encode() not in command
            or b"validation" not in command or (child_proc / "cwd").resolve() != ROOT):
        raise ValueError("Child is not the exact existing CPU validation collector")
    if queue.gpu_state()["compute_pids"]:
        raise ValueError("An active neural process prevents this passive handoff")
    for name in ("smol_precision", "rdt_precision", "rdt_reload", "smol_actual_input_parity"):
        if json.loads((previous / (name + ".json")).read_text()).get("passed") is not True:
            raise ValueError("Finalized model gate missing: " + name)
    return state, child


def launch(output):
    if output.exists():
        raise FileExistsError("Fresh immutable cycle required")
    state, child = validate_handoff()
    output.mkdir(parents=True)
    cross = ROOT / ("VLA/audits/cross_vla120_" + output.name)
    frozen = {str(p): sha256_file(p) for p in (
        Path(__file__).resolve(), ROOT / "VLA/analyze_cross_vla120.py",
        ROOT / "VLA/run_cross_vla120.py", ROOT / "VLA/run_simstate_repair_cycle.py",
        ROOT / "VLA/smolvla/train_simstate.py", ROOT / "VLA/simstate_feedback.py")}
    atomic_json(output / "plan.json", {"previous_cycle": str(PREVIOUS), "previous_snapshot": state,
        "validation_pid": child, "validation_projection": str(PREVIOUS / "validation"),
        "train_projection": str(TRAIN_PROJECTION), "cross_root": str(cross),
        "smolvla_checkpoint": str(SMOL_FINAL), "rdt_checkpoint": str(RDT_FINAL),
        "priority": ["smolvla120", "rdt120", "qualified_simstate_fit_and120", "other_four120_each"],
        "signals_sent_to_coordinators": 1, "training_evaluation_or_replay_processes_signalled": 0,
        "same_robot_objects_cameras_controller_success_seeds": True,
        "checkpoint_selection_NOT_based_on_results": True, "frozen_files": frozen,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    # Sole child is the verified CPU replay. It retains its original stdout
    # descriptor and output; only its passive parent is replaced.
    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while pid_alive(state["pid"]):
        if time.monotonic() > deadline:
            raise TimeoutError("Passive coordinator did not stop")
        time.sleep(.1)
    if not pid_alive(child) and not (PREVIOUS / "validation/audit.json").exists():
        raise ValueError("Adopted replay vanished without a completion receipt")
    with (output / "driver.log").open("x") as log:
        p = subprocess.Popen([str(SIM), "-m", "VLA.run_priority_postfit_cycle", "--run", "--output", str(output)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    atomic_json(output / "launch.json", {"pid": p.pid, "detached_session": True,
        "adopted_validation_pid": child, "checkpoint_weights_not_refit": True})
    print(json.dumps({"pid": p.pid, "output": str(output), "cross_root": str(cross)}, indent=2))


def run(output):
    plan = json.loads((output / "plan.json").read_text())
    lock = (ROOT / "VLA/logs" / (output.name + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR": "1", "VLA_SMOLVLA_INITIAL_FRAME_PROB": ".1",
        "VLA_SMOLVLA_TARGET_BALANCED": "1", "VLA_SMOLVLA_GROUNDING_PROB": "0",
        "VLA_SMOLVLA_SAMPLING_SOURCE": str(SOURCE / "qualified_train")}

    def save(stage, status="running", error=None):
        atomic_json(output / "status.json", {"pid": os.getpid(), "status": status, "stage": stage,
            "error": error, "history": history, "cross_root": plan["cross_root"],
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "no_training_evaluation_or_replay_job_signalled": True,
            "learned_success_established": False, "privileged_inputs_reported_separately": True})

    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in plan["frozen_files"].items()):
            raise ValueError("Frozen new orchestration changed; preserve results")

    def stage(name, command, extra=None, report=None):
        verify(); save(name)
        with (output / (name + ".log")).open("x") as log:
            p = subprocess.Popen(list(map(str, command)), cwd=ROOT, env={**env, **(extra or {})},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / "active_child.json", {"stage": name, "pid": p.pid, "command": list(map(str, command))})
            deadline = time.monotonic() + 86400
            while p.poll() is None:
                save(name)
                if time.monotonic() > deadline:
                    raise TimeoutError("Stage deadline expired; child/evidence preserved for inspection")
                time.sleep(15)
        history.append({"stage": name, "exit_code": p.returncode})
        if p.returncode:
            raise RuntimeError(name + " failed; retained evidence, no unchanged retry")
        if report and json.loads(Path(report).read_text()).get("passed") is not True:
            raise ValueError("Actual gate failed: " + str(report))

    def privileged_cycle():
        validation = Path(plan["validation_projection"])
        deadline = time.monotonic() + 7200
        while pid_alive(plan["validation_pid"]):
            save("await_existing_CPU_validation_receipt")
            if time.monotonic() > deadline:
                raise TimeoutError("CPU validation incomplete; no blind refit")
            time.sleep(15)
        for folder, expected in ((TRAIN_PROJECTION, 2107), (validation, 477)):
            audit = json.loads((folder / "audit.json").read_text())
            if audit.get("passed") is not True or audit.get("source_episodes") != expected:
                raise ValueError("Complete causal replay gate missing")
        from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
        validate_holdout(SOURCE / "qualified_train", SOURCE / "validation_candidate")
        stage("full_actual_loader", [SIM, "-m", "VLA.smolvla.audit_simstate_loader", "--projection", TRAIN_PROJECTION,
            "--dataset", SOURCE / "lerobot_dataset", "--base-model", SMOL_FINAL,
            "--output", output / "full_actual_loader.json"], report=output / "full_actual_loader.json")
        gpu = queue.gpu_state()
        if gpu["compute_pids"] or gpu["free_mib"] < 11000:
            raise ValueError("GPU not quiet at the serialized fit boundary")
        from VLA.run_simstate_repair_cycle import training_command, finalize_checkpoint
        for label, updates, accumulation in (("simstate_smoke", 2, 2), ("simstate_train", 5000, 8)):
            fit = ROOT / "VLA/smolvla_simstate" / output.name / label
            command = training_command(SMOL_FINAL, TRAIN_PROJECTION, fit, updates, accumulation)
            command = [f"--save_freq={min(updates,250)*accumulation}" if str(v).startswith("--save_freq=") else v for v in command]
            stage(label, command, {"VLA_SIMSTATE_TRAIN_PROJECTION": str(TRAIN_PROJECTION),
                "VLA_SIMSTATE_OPTIMIZER_UPDATES": str(updates), "VLA_SMOLVLA_MASTER_AUDIT": str(fit / "master_dtype_audit.json"),
                "VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB": "10000"})
            cp = finalize_checkpoint(fit, updates, accumulation, TRAIN_PROJECTION)
            stage(label + "_precision", [SIM, "-m", "VLA.smolvla.audit_training_precision", "--checkpoint", cp,
                "--output", output / (label + "_precision.json")], report=output / (label + "_precision.json"))
            if json.loads((output / (label + "_precision.json")).read_text())["optimizer_steps"] != [float(updates)]:
                raise ValueError("Saved optimizer count differs")
            stage(label + "_fingerprint", [SIM, "-m", "VLA.fingerprint_checkpoint", cp])
            if updates == 2:
                stage("new_simstate_columns", [SIM, "-m", "VLA.smolvla.audit_simstate_updates", "--base", SMOL_FINAL,
                    "--checkpoint", cp, "--output", output / "new_simstate_columns.json"], report=output / "new_simstate_columns.json")
        stage("simstate120", [SIM, "-m", "VLA.evaluate_simstate120", "--checkpoint", cp, "--output", output / "development120"])
        stage("simstate120_failure_analysis", [SIM, "-m", "VLA.analyze_simstate120",
            "--progress", output / "development120/progress.json", "--output", output / "post120_failure_audit"])

    finished = []
    def after_model(spec, summary):
        verify()
        if summary["status"] == "completed120":
            from VLA.analyze_cross_vla120 import analyze
            report = analyze(Path(plan["cross_root"]) / spec["model"] / "summary.json",
                output / (spec["model"] + "_post120_failure_audit"))
            history.append({"stage": spec["model"] + "120", "completed_trials": 120,
                "successes": report["successes"], "failure_signatures_overlap": report["failure_signatures_overlap"]})
        else:
            history.append({"stage": spec["model"] + "120", "integration_incomplete": True, "error": summary["error"]})
        finished.append(spec["model"])
        if finished == ["smolvla", "rdt"]:
            # This variant was declared BEFORE outcomes; it is not chosen by SR.
            try:
                privileged_cycle()
            except Exception as error:
                history.append({"stage": "privileged_cycle", "preserved_error": f"{type(error).__name__}: {error}"})
                save("privileged_error_preserved_other_models_continue")
        save("standard120_queue_continue")

    try:
        verify(); save("initialize_same_scene_priority120_queue")
        original = queue.model_specs
        def specs():
            rows = original()
            for row in rows:
                if row["model"] == "rdt":
                    row["checkpoint"] = str(RDT_FINAL)
                    row["continued_from_saved_R16_optimizer_updates"] = 1000
                    row["DataLoader_shuffle_restarted_NOT_bit_exact"] = True
            order = {name: i for i, name in enumerate(("smolvla", "rdt", "pi0", "pi0_fast", "openvla", "cogact"))}
            return sorted(rows, key=lambda row: order[row["model"]])
        queue.model_specs = specs
        cross = Path(plan["cross_root"])
        cross_plan = queue.initialize(cross)
        cross_plan["checkpoint_selection_rule"] = "Saved declared Smol/RDT finals; official bases otherwise, NOT outcome-selected"
        atomic_json(cross / "plan.json", cross_plan)
        save("standard120_queue_smolvla_then_rdt")
        queue.run_queue(cross, cross_plan, 72, after_model=after_model)
        data = json.loads((cross / "progress.json").read_text())
        incomplete = data["status"] != "completed720" or any("preserved_error" in h for h in history)
        save("measured_cohorts_REQUIRE_actual_SR_and_video_review", status="finished_with_explicit_incomplete_lanes" if incomplete else "completed")
    except Exception as error:
        save("preserved_error_NO_false_success", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true"); mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch(args.output.resolve()) if args.launch else run(args.output.resolve())
