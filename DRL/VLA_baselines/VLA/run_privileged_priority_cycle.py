"""Prioritize audited causal simulator inputs after the user's new request.

Completed baseline episodes are never rerolled. A verified passive queue
owner is replaced, its in-flight RDT episode block is allowed to finish,
and its owned server is closed only at that episode boundary. Native VLA
actions remain the ONLY action producer in the separately scored state18 row.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA import run_cross_vla120 as queue
from VLA.recover_interrupted_cycles import ROOT, SIM, SOURCE, SMOL_FINAL, pid_alive

PREVIOUS = ROOT / "VLA/audits/priority_postfit_r20_20261014"
CROSS = ROOT / "VLA/audits/cross_vla120_priority_postfit_r20_20261014"
PROJECTION = ROOT / "VLA/audits/recovery_r17_20261013/train"
VALIDATION = ROOT / "VLA/audits/recovery_r19_metadata_gate_20261014/validation"


def command_tokens(pid):
    return (Path(f"/proc/{pid}") / "cmdline").read_bytes().split(b"\0")


def check_owner(pid):
    proc = Path(f"/proc/{pid}")
    tokens = command_tokens(pid)
    if (not pid_alive(pid) or b"VLA.run_priority_postfit_cycle" not in tokens
            or str(PREVIOUS).encode() not in tokens or (proc / "cwd").resolve() != ROOT):
        raise ValueError("Not the exact previous queue owner")


def check_frozen_snapshot(state, progress, children, tokens_by_pid, gpu_pids, old_plan):
    """Pure handoff preconditions: NEVER accept a live training child."""
    if (state.get("status") != "running" or progress.get("status") != "running"
            or progress.get("pid") != state.get("pid") or progress.get("current_model") != "rdt"):
        raise ValueError("Only the declared ordinary RDT evaluation wait is eligible")
    seed = progress.get("current_seed")
    if seed not in queue.SEEDS or len(children) not in (1, 2):
        raise ValueError("Unexpected current seed/children")
    spec = next(r for r in old_plan["models"] if r["model"] == "rdt")
    servers, trials = [], []
    trial_path = ROOT / "VLA/rdt" / CROSS.name / f"test_seed{seed}"
    for pid in children:
        tokens = tokens_by_pid[pid]
        if b"VLA.rdt.server" in tokens and spec["checkpoint"].encode() in tokens and b"5720" in tokens:
            servers.append(pid)
        elif (b"VLA.cross_vla120_trial" in tokens and str(trial_path).encode() in tokens
                and b"rdt" in tokens and str(seed).encode() in tokens):
            trials.append(pid)
        else:
            raise ValueError("Unexpected child (possibly training); do not hand off")
    if len(servers) != 1 or len(trials) > 1 or set(map(int, gpu_pids)) != set(servers):
        raise ValueError("GPU/server identity differs; no neural process is signalled")
    return {"server_pid": servers[0], "trial_pid": trials[0] if trials else None,
        "current_seed": seed, "trial_path": str(trial_path), "rdt_spec": spec}


def qualify_existing_data():
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
    from VLA.common.source_training_gate import require_training_source
    require_training_source(SOURCE / "qualified_train", model="smolvla", optimizer_updates=5000)
    validate_holdout(SOURCE / "qualified_train", SOURCE / "validation_candidate")
    reports = {}
    for folder, count, frames in ((PROJECTION, 2107, 116257), (VALIDATION, 477, 26837)):
        doc = json.loads((folder / "audit.json").read_text())
        if (doc.get("passed") is not True or doc.get("source_episodes") != count or doc.get("frames") != frames
                or doc.get("future_state_in_model_input") is not False or doc.get("teacher_actions_in_model_input") is not False):
            raise ValueError("Full causal TRAIN/validation receipt missing")
        reports[str(folder / "audit.json")] = sha256_file(folder / "audit.json")
    loader = PREVIOUS / "preflight_full_actual_loader_now.json"
    doc = json.loads(loader.read_text())
    if (doc.get("passed") is not True or doc.get("samples") != 128
            or doc.get("state18_normalization_max_abs_error") != 0
            or doc.get("projection_audit_sha256") != reports[str(PROJECTION / "audit.json")]
            or doc.get("stats_sha256") != sha256_file(PROJECTION / "stats.json")):
        raise ValueError("Actual full-source loader/normalizer gate missing or stale")
    reports[str(loader)] = sha256_file(loader)
    return reports


def repair_plan(previous, output):
    """Only the diagnosed pre-update batch-dispatch failure may be repaired."""
    old = json.loads((previous / "plan.json").read_text())
    state = json.loads((previous / "status.json").read_text())
    child = json.loads((previous / "active_child.json").read_text())
    log = (previous / "simstate_smoke.log").read_text()
    if (state.get("status") != "failed" or pid_alive(state.get("pid")) or pid_alive(child.get("pid"))
            or child.get("stage") != "simstate_smoke"
            or "The size of tensor a (7) must match the size of tensor b (18)" not in log
            or "_preprocess_dataset_batch" not in log
            or (ROOT / "VLA/smolvla_simstate" / previous.name / "simstate_smoke/checkpoints").exists()
            or not (previous / "preserved_baseline_prefix.json").exists()):
        raise ValueError("Not the preserved PRE-update state7 batched-dispatch failure")
    batch = previous / "fixed_actual_training_batches.json"
    doc = json.loads(batch.read_text())
    if (doc.get("passed") is not True or doc.get("samples") != 128 or doc.get("actual_batches") != 64
            or doc.get("torch_DataLoader_batched_fetch_exercised") is not True
            or doc.get("state18_batch_normalization_max_abs_error") != 0
            or any(v != 0 for v in doc.get("ordinary_state7_native_batch_vs_indexed_max_errors",{}).values())
            or doc.get("wrapper_sha256") != sha256_file(ROOT / "VLA/smolvla/simstate_dataset.py")):
        raise ValueError("Actual temporal DataLoader fix is not qualified")
    if queue.gpu_state()["compute_pids"]: raise ValueError("GPU has an unknown owner")
    old.update(repair_from_failed_cycle=str(previous), baseline_boundary_completed=True,
        diagnosed_fix="Override __getitems__: Torch DataLoader must not bypass the state18 overlay",
        pre_update_failure_not_a_training_result=True,
        previous_failed_status=state, coordinator_signals=[], active_episode_processes_signalled=0,
        training_processes_signalled=0, existing_data_proofs=qualify_existing_data())
    old["existing_data_proofs"][str(batch)] = sha256_file(batch)
    old["existing_data_proofs"][str(previous / "preserved_baseline_prefix.json")] = sha256_file(previous / "preserved_baseline_prefix.json")
    old["frozen_files"] = {str(p):sha256_file(p) for p in (Path(__file__).resolve(),
        ROOT / "VLA/run_simstate_repair_cycle.py", ROOT / "VLA/simstate_feedback.py",
        ROOT / "VLA/smolvla/train_simstate.py", ROOT / "VLA/smolvla/simstate_dataset.py",
        ROOT / "VLA/smolvla/simstate_adapter.py", ROOT / "VLA/evaluate_simstate120.py")}
    old["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", old)
    return old


def launch(output, repair_from=None):
    if output.exists(): raise FileExistsError("New immutable cycle required")
    if repair_from is not None:
        repair_plan(repair_from.resolve(),output)
        with (output / "driver.log").open("x") as log:
            child=subprocess.Popen([str(SIM),"-m","VLA.run_privileged_priority_cycle","--run","--output",str(output)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(output / "launch.json",{"pid":child.pid,"detached_session":True,
            "diagnosed_pre_update_failure_fixed":True,"no_old_process_signalled":True})
        print(json.dumps({"pid":child.pid,"output":str(output)},indent=2));return
    reports = qualify_existing_data()
    old_plan = json.loads((CROSS / "plan.json").read_text()); queue.verify_plan(old_plan)
    state = json.loads((PREVIOUS / "status.json").read_text())
    owner = state["pid"]; check_owner(owner)
    frozen = {str(p): sha256_file(p) for p in (Path(__file__).resolve(),
        ROOT / "VLA/run_simstate_repair_cycle.py", ROOT / "VLA/simstate_feedback.py",
        ROOT / "VLA/smolvla/train_simstate.py", ROOT / "VLA/smolvla/simstate_dataset.py",
        ROOT / "VLA/smolvla/simstate_adapter.py", ROOT / "VLA/evaluate_simstate120.py")}
    # Freeze ONLY the verified passive owner while checking its children.
    # This eliminates the previous-seed/next-seed handoff race. Neural child
    # processes are not stopped and continue their current episode block.
    os.kill(owner, signal.SIGSTOP)
    handed_off = False
    try:
        progress = json.loads((CROSS / "progress.json").read_text())
        children = list(map(int, (Path(f"/proc/{owner}/task/{owner}/children")).read_text().split()))
        info = check_frozen_snapshot(state, progress, children,
            {pid: command_tokens(pid) for pid in children}, queue.gpu_state()["compute_pids"], old_plan)
        output.mkdir(parents=True)
        atomic_json(output / "plan.json", {"previous_cycle": str(PREVIOUS), "cross_root": str(CROSS),
            "previous_state": state, "previous_progress": progress, "handoff": info,
            "existing_data_proofs": reports, "frozen_files": frozen,
            "base_model": str(SMOL_FINAL), "train_projection": str(PROJECTION), "validation_projection": str(VALIDATION),
            "privileged_model_inputs": True, "oracle_controller_at_deployment": False,
            "new_optimizer_updates": 5000, "prior_base_family_updates": 15000,
            "ordinary_baseline_deferred_NOT_rerolled_or_scored_zero": True,
            "user_priority": "more causal simulator information to improve learned VLA control",
            "coordinator_signals": ["SIGSTOP", "SIGTERM", "SIGCONT"],
            "active_episode_processes_signalled": 0, "training_processes_signalled": 0,
            "created_at_utc": datetime.now(timezone.utc).isoformat()})
        os.kill(owner, signal.SIGTERM)
        os.kill(owner, signal.SIGCONT)
        handed_off = True
        deadline = time.monotonic() + 15
        while pid_alive(owner):
            if time.monotonic() > deadline: raise TimeoutError("Previous passive coordinator did not stop")
            time.sleep(.1)
    finally:
        if not handed_off and pid_alive(owner): os.kill(owner, signal.SIGCONT)
    with (output / "driver.log").open("x") as log:
        child = subprocess.Popen([str(SIM), "-m", "VLA.run_privileged_priority_cycle", "--run", "--output", str(output)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    atomic_json(output / "launch.json", {"pid": child.pid, "detached_session": True,
        "inflight_RDT_block_keeps_running": True, "no_duplicate_state18_fit": True})
    print(json.dumps({"pid": child.pid, "output": str(output)}, indent=2))


def merge_committed_rdt_prefix(progress, plan, current_result):
    """Adopt only completed prefix receipts; never invent/reroll an episode."""
    spec = plan["handoff"]["rdt_spec"]
    old = progress["models"]["rdt"]
    rows = list(old["rows"])
    seed = plan["handoff"]["current_seed"]
    if (current_result.get("model") != "rdt" or current_result.get("seed") != seed
            or current_result.get("integration_error") is not None
            or {t["name"] for t in current_result["tasks"]} != set(queue.TASKS)
            or len(current_result["tasks"]) != 4
            or any(t.get("error") or not isinstance(t["success"], bool) for t in current_result["tasks"])):
        raise ValueError("In-flight block did not finish validly; preserve incomplete evidence")
    current = {(r["seed"], r["task"]) for r in rows}
    block = {(seed, t) for t in queue.TASKS}
    if current & block and not block <= current:
        raise ValueError("Partially committed block cannot be silently merged")
    if not block <= current:
        folder = Path(plan["handoff"]["trial_path"])
        rows.extend({"seed": seed, "task": t["name"], "success": t["success"], "error": None,
            "safety_violation": t.get("safety_violation"), "result": str(folder / "results.json"),
            "trajectory": str(folder / "trajectory.json"), "video": str(folder / "rollout.mp4") if seed == queue.SEEDS[0] else None}
            for t in current_result["tasks"])
    expected = {(s, t) for s in queue.SEEDS[:queue.SEEDS.index(seed)+1] for t in queue.TASKS}
    if len(rows) != len(expected) or {(r["seed"], r["task"]) for r in rows} != expected:
        raise ValueError("Committed prefix has missing/duplicate/undeclared episodes")
    return queue.summarize_model({}, spec, rows, "deferred_by_user_privileged_input_priority")


def run(output):
    plan = json.loads((output / "plan.json").read_text())
    lock = (ROOT / "VLA/logs" / (output.name + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR": "1", "VLA_SMOLVLA_INITIAL_FRAME_PROB": ".1",
        "VLA_SMOLVLA_TARGET_BALANCED": "1", "VLA_SMOLVLA_GROUNDING_PROB": "0",
        "VLA_SMOLVLA_SAMPLING_SOURCE": str(SOURCE / "qualified_train"), "VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB": "10000"}

    def save(stage, status="running", error=None):
        atomic_json(output / "status.json", {"pid": os.getpid(), "status": status, "stage": stage,
            "error": error, "history": history, "cross_root": str(CROSS),
            "privileged_model_inputs": True, "oracle_controller_at_deployment": False,
            "new_optimizer_updates": 5000, "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "plain_VLA_success_NOT_claimed": True, "stable_high_success_NOT_established": True})

    def verify():
        for p, digest in {**plan["frozen_files"], **plan["existing_data_proofs"]}.items():
            if sha256_file(Path(p)) != digest: raise ValueError("Qualified source/implementation changed")

    def stage(name, command, extra=None, report=None):
        verify(); save(name)
        with (output / (name + ".log")).open("x") as log:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env={**env, **(extra or {})},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / "active_child.json", {"stage": name, "pid": child.pid, "command": list(map(str, command))})
            deadline = time.monotonic() + 86400
            while child.poll() is None:
                save(name)
                if time.monotonic() > deadline: raise TimeoutError("Stage deadline; child/evidence preserved")
                time.sleep(10)
        history.append({"stage": name, "exit_code": child.returncode})
        if child.returncode: raise RuntimeError(name + " failed; no unchanged retries")
        if report and json.loads(Path(report).read_text()).get("passed") is not True:
            raise ValueError("Actual gate failed: " + str(report))

    try:
        verify(); save("allow_inflight_RDT_episode_block_to_finish")
        inflight = None if plan.get("baseline_boundary_completed") else plan["handoff"]["trial_pid"]
        deadline = time.monotonic() + 7200
        while inflight and pid_alive(inflight):
            save("allow_inflight_RDT_episode_block_to_finish")
            if time.monotonic() > deadline: raise TimeoutError("In-flight episode unfinished; no neural child killed")
            time.sleep(10)
        folder = Path(plan["handoff"]["trial_path"])
        result = json.loads((folder / "results.json").read_text())
        progress = plan["previous_progress"]
        progress["models"]["rdt"] = merge_committed_rdt_prefix(progress, plan, result)
        prefix = progress["models"]["rdt"]
        evidence = {}
        for row in prefix["rows"]:
            for name in (row["result"], row["trajectory"], str(Path(row["result"]).parent / "manifest.json")):
                evidence[name] = sha256_file(Path(name))
        atomic_json(output / "preserved_baseline_prefix.json", {"rows": prefix["rows"], "evidence": evidence,
            "current_inflight_receipt_adopted": True, "episodes_rerolled": 0, "standard_model": "rdt"})
        progress.update(status="deferred_by_user_privileged_input_priority", pid=os.getpid(),
            priority_cycle=str(output), updated_at_utc=datetime.now(timezone.utc).isoformat())
        progress["completed_neural_trials"] = sum(s["completed_trials"] for s in progress["models"].values())
        atomic_json(CROSS / "progress.json", progress)
        atomic_json(CROSS / "rdt/summary.json", prefix)
        from VLA.report_cross_vla120 import render
        (CROSS / "REPORT.md").write_text(render(progress))
        server = None if plan.get("baseline_boundary_completed") else plan["handoff"]["server_pid"]
        if pid_alive(server):
            tokens = command_tokens(server)
            if b"VLA.rdt.server" not in tokens or plan["handoff"]["rdt_spec"]["checkpoint"].encode() not in tokens:
                raise ValueError("Server identity changed; do not signal")
            os.kill(server, signal.SIGTERM)  # AFTER its episode block finishes.
            deadline = time.monotonic() + 30
            while pid_alive(server):
                if time.monotonic() > deadline: raise TimeoutError("Owned idle server did not stop; no fit overlap")
                time.sleep(.2)
        history.append({"stage": "baseline_boundary_handoff", "rdt_completed_trials": prefix["completed_trials"],
            "server_closed_after_episode_NOT_mid_episode": True, "episodes_rerolled": 0})
        gpu = queue.gpu_state()
        if gpu["compute_pids"] or gpu["free_mib"] < 11000:
            raise ValueError("GPU not quiet; do not overlap an unknown owner")
        from VLA.fingerprint_checkpoint import verify_fingerprint
        verify_fingerprint(SMOL_FINAL, full=True)
        stage("full_actual_loader", [SIM, "-m", "VLA.smolvla.audit_simstate_loader", "--projection", PROJECTION,
            "--dataset", SOURCE / "lerobot_dataset", "--base-model", SMOL_FINAL,
            "--output", output / "full_actual_loader.json"], report=output / "full_actual_loader.json")
        from VLA.run_simstate_repair_cycle import training_command, finalize_checkpoint
        for label, updates, accumulation in (("simstate_smoke", 2, 2), ("simstate_train", 5000, 8)):
            fit = ROOT / "VLA/smolvla_simstate" / output.name / label
            command = training_command(SMOL_FINAL, PROJECTION, fit, updates, accumulation)
            command = [f"--save_freq={min(updates,250)*accumulation}" if str(v).startswith("--save_freq=") else v for v in command]
            stage(label, command, {"VLA_SIMSTATE_TRAIN_PROJECTION": str(PROJECTION),
                "VLA_SIMSTATE_OPTIMIZER_UPDATES": str(updates), "VLA_SMOLVLA_MASTER_AUDIT": str(fit / "master_dtype_audit.json")})
            cp = finalize_checkpoint(fit, updates, accumulation, PROJECTION)
            stage(label + "_precision", [SIM, "-m", "VLA.smolvla.audit_training_precision", "--checkpoint", cp,
                "--output", output / (label + "_precision.json")], report=output / (label + "_precision.json"))
            if json.loads((output / (label + "_precision.json")).read_text())["optimizer_steps"] != [float(updates)]:
                raise ValueError("Saved optimizer count differs")
            stage(label + "_fingerprint", [SIM, "-m", "VLA.fingerprint_checkpoint", cp])
            if updates == 2:
                stage("new_simstate_columns", [SIM, "-m", "VLA.smolvla.audit_simstate_updates", "--base", SMOL_FINAL,
                    "--checkpoint", cp, "--output", output / "new_simstate_columns.json"], report=output / "new_simstate_columns.json")
        stage("privileged120", [SIM, "-m", "VLA.evaluate_simstate120", "--checkpoint", cp, "--output", output / "development120"])
        stage("privileged120_failure_analysis", [SIM, "-m", "VLA.analyze_simstate120", "--progress", output / "development120/progress.json",
            "--output", output / "post120_failure_audit"])
        # The original RDT prefix and four other declared models are retained
        # for an explicit no-reroll continuation, not silently considered done.
        save("privileged120_measured_standard_cohort_resume_pending", status="completed_privileged_cycle_standard_cohort_not_complete")
    except Exception as error:
        save("preserved_error_NO_false_success", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--launch", action="store_true"); modes.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-from",type=Path,help="Explicit fresh repair of the diagnosed zero-update batch-dispatch failure")
    args = parser.parse_args()
    if args.repair_from and not args.launch: raise ValueError("--repair-from only applies to a fresh launch")
    launch(args.output.resolve(),args.repair_from) if args.launch else run(args.output.resolve())
