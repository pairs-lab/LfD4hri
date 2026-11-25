"""User-revised40 baseline -> error analysis -> audited fit ->40 postfit.

Reuse completed CPU preparation without restarting conversion/replay. Stop
only the identified old continuation at its CPU-ready waiting barrier. All
models/scenes/safety remain unchanged; failures are never filtered or rerolled.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.run_smolvla_v9_pilot import parse_args, checkpoint_path, validate_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "baseline-progress", "output", "model-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prep, output = args.preparation.resolve(), args.output.resolve()
    baseline = args.baseline_progress.resolve()
    if output.exists() or args.model_output.exists():
        raise FileExistsError("Fresh cycle evidence/model experiment required")
    previous = json.loads((prep / "plan.json").read_text())
    fixed = {str(p): sha256_file(p) for p in (Path(__file__).resolve(), ROOT / "VLA/run_diagnostic40.py",
        ROOT / "VLA/analyze_diagnostic40.py", ROOT / "VLA/analyze_diagnostic50.py", prep / "plan.json",
        baseline.parent / "plan.json")}
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    history = []
    contract = {"scope": "user_requested_40_trial_DEVELOPMENT_audit_fix_fit_test_cycle",
        "baseline_progress": str(baseline), "reuse_CPU_preparation": str(prep),
        "model_output": str(args.model_output.resolve()), "trials_per_action": 10,
        "trials_each_evaluation": 40, "updates": previous["updates"],
        "frozen_files": fixed, "final_test_used": False, "oracle_fallback": False,
        "high_or_stable_success_guaranteed": False, "old200_wait_replaced_by_user_request": True}
    (output / "plan.json").write_text(json.dumps(contract, indent=2) + "\n")
    def save(stage, error=None):
        (output / "status.json").write_text(json.dumps({**contract, "status": stage, "pid": os.getpid(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "history": history, "error": error}, indent=2) + "\n")
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in fixed.items()):
            raise ValueError("Fixed preparation/evaluation/cycle implementation changed")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    def run(stage, module, tokens):
        verify()
        save(stage)
        command = [sys.executable, "-m", module, *map(str, tokens)]
        with (output / "logs" / (stage + ".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"stage": stage, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{stage} failed; diagnose exact preserved evidence")
        save(stage + "_completed")
    try:
        deadline = time.monotonic() + 86400
        analyzed = False
        while True:
            verify()
            b = json.loads(baseline.read_text())
            state = json.loads((prep / "status.json").read_text())
            if b["status"].startswith("stopped") or state["status"].startswith("stopped"):
                raise ValueError(f"Baseline/preparation failure: {b.get('error')} / {state.get('error')}")
            if b["status"] == "completed40_development_trials" and not analyzed:
                run("baseline40_error_analysis", "VLA.analyze_diagnostic40", ["--progress", baseline,
                    "--output", output / "baseline40_analysis"])
                analyzed = True
                summary = json.loads((output / "baseline40_analysis/analysis.json").read_text())
                (output / "fix_evidence_and_limits.json").write_text(json.dumps({
                    "baseline_failure_signatures": summary["failure_signatures_overlap"],
                    "applied_repairs": summary["repairs_applied_before_new_fit"],
                    "unresolved_limits": summary["limitations"], "training_on_evaluation_labels": False,
                    "setup_causes_exhaustively_proven": False, "old_checkpoint_did_not_use_repaired_source": True,
                    "data_sufficiency_established": False}, indent=2) + "\n")
            if analyzed and state["status"] == "prepared_CPU_waiting_complete200_baseline":
                break
            save("waiting_baseline40_and_prepared_CPU_no_GPU_overlap")
            if time.monotonic() >= deadline:
                raise TimeoutError("Finite baseline/preparation wait expired")
            time.sleep(20)
        required = {"materialize_train", "materialize_validation", "train_candidate_audit", "validation_audit",
            "qualify", "qualified_train_audit", "sampler_phase_mass", "cpu_convert", "cpu_exact_stats",
            "cpu_roundtrip", "cpu_preprocess", "cpu_validate", "cpu_train_reset_sensor_parity", "cpu_heldout_reset_sensor_parity"}
        successful = {h["stage"] for h in state["history"] if h["exit_code"] == 0}
        if not required <= successful:
            raise ValueError("Completed immutable CPU preparation missing required stages")
        for name in ("stats.json", "roundtrip.json", "smolvla_preprocess.json"):
            p = prep / "fit_audit/pilot_prep" / name
            if not json.loads(p.read_text())["passed"]:
                raise ValueError("Failed reusable CPU preparation")
            fixed[str(p)] = sha256_file(p)
        tokens = ["--source", prep / "qualified_train", "--validation", prep / "validation_candidate",
            "--source-audit", prep / "qualified_train_audit/audit.json", "--validation-audit", prep / "validation_audit/audit.json",
            "--dataset-root", prep / "lerobot_dataset", "--prep-root", prep / "fit_audit/pilot_prep",
            "--audit-root", output / "fit_audit", "--output", args.model_output, "--updates", previous["updates"],
            "--experiment-scope", "bounded_refinement_experiment", "--batch-size", previous["batch_size"],
            "--learning-rate", previous["learning_rate"],
            "--initial-frame-sampling-probability", previous["initial_frame_sampling_probability"], "--prepared"]
        if previous["paired_grounding_probability"]:
            tokens += ["--paired-grounding-probability", previous["paired_grounding_probability"]]
        pilot = parse_args(list(map(str, tokens)))
        validate_inputs(pilot)
        pid = int(state["pid"])
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if (b"VLA.resume_readiness_corrected_fit" not in argv
                or os.readlink(f"/proc/{pid}/cwd") != str(ROOT)
                or Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()):
            raise ValueError("Old job is not the identified child-free CPU-ready waiter")
        os.kill(pid, signal.SIGTERM)
        (output / "preparation_handoff.json").write_text(json.dumps({
            "old_waiter_pid_terminated": pid, "old_status_snapshot": state,
            "reason": "User changed200 trials to40; completed CPU/source/replay proof preserved",
            "no_payload_deleted_or_replayed": True, "new_source": str(pilot.source)}, indent=2) + "\n")
        run("finetune_and_source_diagnostics", "VLA.run_smolvla_v9_pilot", [*tokens, "--run"])
        cp = checkpoint_path(args.model_output / "train", previous["updates"], pilot.accumulation)
        run("postfit40_development", "VLA.run_diagnostic40", ["--checkpoint", cp,
            "--output", output / "postfit40", "--run-tag", "diagnostic40_r14_postfit_20261013"])
        run("postfit40_error_analysis", "VLA.analyze_diagnostic40", ["--progress", output / "postfit40/progress.json",
            "--output", output / "postfit40_analysis"])
        report = json.loads((output / "postfit40_analysis/analysis.json").read_text())
        b_report = json.loads((output / "baseline40_analysis/analysis.json").read_text())
        result = {"baseline_successes": b_report["successes"], "postfit_successes": report["successes"],
            "trials_each": 40, "per_action": report["statistics"]["by_task"],
            "failure_signatures": report["failure_signatures_overlap"], "final_test_used": False,
            "stable_success_established": False,
            "followup_required": "Root must inspect actual postfit video/trace and fix remaining causes; do not repeat unchanged training blindly."}
        (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        save("completed40_fit_test_cycle_remaining_errors_require_review")
    except Exception as exc:
        save("stopped_no_silent_retry", str(exc))
        raise


if __name__ == "__main__":
    main()
