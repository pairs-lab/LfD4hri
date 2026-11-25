"""Finite continuation after repaired-source CPU gates and ACTUAL root review.

Never executes commands from a JSON document. Constructs the pinned, scoped
qualification/conversion/training commands itself. Stops on every exception.
No review receipt is fabricated, no scene is skipped, no failed fit is retried.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.prepare_v9_candidate_fit import require_visual_receipt

ROOT = Path(__file__).resolve().parents[1]


def ready_for_reviewed_continuation(workflow):
    workflow = Path(workflow)
    status = json.loads((workflow / "workflow_status.json").read_text())
    if status["status"].startswith(("stopped", "interrupted")):
        raise ValueError("CPU preparation failed/interrupted; no fit")
    if status["status"] != "awaiting_actual_final_source_visual_review_no_training":
        return False
    receipt = workflow / "visual_review_receipt.json"
    if not receipt.exists():
        return False
    require_visual_receipt(receipt, Path(status["candidate"]), workflow / "native_smol512",
                           workflow / "all_repaired_train_audit/audit.json")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=14400)
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    status_path = workflow / "resume_status.json"
    if status_path.exists() or not 1 <= args.wait_seconds <= 14400:
        raise ValueError("Fresh finite continuation required")
    plan_path = workflow / "workflow_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["scope"] != "bounded_refinement_preparation_NOT_final_benchmark" or plan["final_test_used"]:
        raise ValueError("Not the scoped repaired TRAIN preparation")
    contract = {"workflow": str(workflow), "workflow_plan_sha256": sha256_file(plan_path),
        "resume_implementation_sha256": sha256_file(Path(__file__)), "run_training_requested": args.run_training,
        "updates": plan["updates"], "final_test_used": False, "gpu_optimizer_started": False}
    python = str(ROOT / "VLA/third_party/lerobot/.venv/bin/python")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "TOKENIZERS_PARALLELISM": "false"}
    history = []
    logs = workflow / "resume_logs"
    logs.mkdir()
    def status(state, stage, error=None):
        status_path.write_text(json.dumps({**contract, "status": state, "stage": stage,
            "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "history": history, "error": error}, indent=2) + "\n")
    def unchanged():
        if (sha256_file(plan_path) != contract["workflow_plan_sha256"]
                or sha256_file(Path(__file__)) != contract["resume_implementation_sha256"]
                or any(sha256_file(Path(p)) != digest for p, digest in plan["frozen_references"].items())):
            raise ValueError("Frozen plan, references or implementation changed")
    def stage(name, module, tokens, timeout=10800):
        unchanged()
        status("running", name)
        command = [python, "-m", module, *map(str, tokens)]
        with (logs / (name + ".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        history.append({"stage": name, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed: {logs / (name + '.log')}; no automatic retry")
        unchanged()
        status("stage_completed", name)
    try:
        deadline = time.monotonic() + args.wait_seconds
        status("waiting_for_complete_cpu_gates_and_actual_review", "wait")
        while not ready_for_reviewed_continuation(workflow):
            unchanged()
            if time.monotonic() >= deadline:
                raise TimeoutError("No completed CPU gates and actual image review within finite wait; no training")
            time.sleep(20)
        source = Path(plan["candidate"])
        train = workflow / "inner_train_candidate"
        qualified = workflow / "qualified_inner_train"
        audit = workflow / "all_repaired_train_audit/audit.json"
        train_audit = workflow / "inner_train_audit/audit.json"
        stage("qualify", "VLA.qualify_bounded_refinement", ["--source", train, "--source-audit", train_audit,
            "--native-replay-report", workflow / "all_repaired_train_full_physics_replay.json", "--output", qualified,
            "--max-updates", plan["updates"], "--visual-review-receipt", workflow / "visual_review_receipt.json",
            "--native-camera-dir", workflow / "native_smol512", "--combined-source", source, "--combined-source-audit", audit])
        qualified_audit = workflow / "qualified_train_audit/audit.json"
        counts = Counter(r["task"]["name"] for r in json.loads((qualified / "index.json").read_text()))
        stage("qualified_audit", "VLA.audit_finetune_data", ["--dataset", qualified, "--output", qualified_audit.parent,
            "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract"])
        stage("final_train_support", "VLA.audit_v9_data_support", ["--source", qualified, "--source-audit", qualified_audit,
            "--output", workflow / "qualified_train_support.json"])
        fit = ["--source", qualified, "--validation", plan["outer_validation"], "--dataset-root", workflow / "lerobot_dataset",
            "--audit-root", workflow / "fit_audit", "--source-audit", qualified_audit,
            "--validation-audit", workflow / "outer_validation_audit/audit.json", "--output", plan["model_output"],
            "--updates", plan["updates"], "--experiment-scope", "bounded_refinement_experiment"]
        stage("fit_plan", "VLA.run_smolvla_v9_pilot", fit)
        if not args.run_training:
            status("qualified_and_planned_no_optimizer", "done")
            return
        # This read-only allocation guard complements the driver's CUDA smoke.
        # It does not stop any other process or wait indefinitely for resources.
        resources = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30)
        rows = [tuple(map(int, row.split(","))) for row in resources.stdout.strip().splitlines()]
        if len(rows) != 1 or rows[0][0] < 12000 or rows[0][1] > 10:
            raise RuntimeError("GPU allocation guard failed; no other workload is stopped")
        # 'started' is recorded by the actual driver per stage, not assumed
        # from launching a command that first converts and audits data.
        contract["gpu_optimizer_started"] = None
        contract["actual_driver_status"] = str(workflow / "fit_audit/pilot_driver_status.json")
        status("running_gated_fit_driver_optimizer_not_yet_confirmed", "fit_driver")
        stage("fit_and_evaluate", "VLA.run_smolvla_v9_pilot", [*fit, "--run"], timeout=172800)
        contract["gpu_optimizer_started"] = True
        status("completed_bounded_fit_and_diagnostic_rollouts", "done")
    except BaseException as exc:
        status("stopped_no_silent_retry", "failure", repr(exc))
        raise


if __name__ == "__main__":
    main()
