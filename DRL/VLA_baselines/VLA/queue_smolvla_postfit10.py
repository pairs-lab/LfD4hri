"""Wait for ONE declared qualified fit, then run the locked 40-trial pilot.

Does not train, select a checkpoint from rollout outcomes, bypass image review,
serve another architecture, or certify a minimum learned success rate.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.run_evaluation_protocol import validate_protocol, locked_split_seeds
from VLA.run_smolvla_v9_pilot import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]


def fit_finished(state, expected_updates):
    if state.get("updates") != expected_updates or not state.get("run_training_requested"):
        raise ValueError("Upstream fit does not match the declared optimization budget")
    if state.get("status") == "stopped_no_silent_retry":
        raise RuntimeError("Upstream workflow stopped: " + str(state.get("error")))
    return state.get("status") == "completed_diagnostic_only"


def ready_checkpoint(workflow, expected_updates):
    fit = workflow / "fit_audit"
    driver = json.loads((fit / "pilot_driver_status.json").read_text())
    if (driver.get("status") != "completed_diagnostic_only"
            or driver.get("updates") != expected_updates
            or driver.get("final_test_used")
            or driver.get("validation_used_for_training_or_normalization")):
        raise RuntimeError("Fit/parity/diagnostic stages are not complete and clean")
    stage = next(s for s in driver["stages"] if s["name"] == "train_precision")
    command = stage["command"]
    checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
    identity = verify_fingerprint(checkpoint, full=True)
    for name in ("train_precision", "train_input_parity", "train_reset_sensor_parity", "heldout_reset_sensor_parity"):
        report = json.loads((fit / (name + ".json")).read_text())
        if not report.get("passed"):
            raise RuntimeError("Required fit proof failed: " + name)
        if name.startswith("train_") and not name.endswith("sensor_parity"):
            if Path(report.get("checkpoint", "")).resolve() != checkpoint:
                raise RuntimeError("Proof refers to another checkpoint")
        if name == "train_precision" and report.get("optimizer_steps") != [float(expected_updates)]:
            raise RuntimeError("Incorrect real optimizer-update count")
        if name == "train_input_parity" and report.get("checkpoint_content_sha256") != identity["content_tree_sha256"]:
            raise RuntimeError("Input parity does not bind the actual completed weights")
    return checkpoint, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--oracle-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--wait-seconds", type=int, default=604800)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.wait_seconds <= 604800:
        raise ValueError("Use a new output and a finite wait of at most seven days")
    protocol, protocol_hash = validate_protocol(args.protocol.resolve(), args.oracle_audit.resolve())
    if (protocol.get("evaluation_scope") != "postfit_10_per_action_pilot"
            or len(locked_split_seeds(protocol, "development")) != 10
            or protocol["tasks"] != ["touch", "pick", "move", "pick_place"]):
        raise ValueError("Requires the locked four-task 10/action pilot")
    workflow = args.workflow.resolve()
    initial = json.loads((workflow / "workflow_status.json").read_text())
    fit_finished(initial, args.updates)
    expected = checkpoint_path(Path(initial["model_output"]) / "train", args.updates, 16).resolve()
    frozen = {str(path.resolve()): sha256_file(path) for path in (
        Path(__file__), ROOT / "VLA/run_evaluation_protocol.py", ROOT / "VLA/benchmark.py",
        args.protocol, args.oracle_audit, workflow / "workflow_plan.json")}
    args.output.mkdir(parents=True)
    state = {"model": "smolvla", "scope": "postfit_10_per_action_pilot_NOT_final_test",
             "expected_checkpoint": str(expected), "expected_optimizer_updates": args.updates,
             "protocol_sha256": protocol_hash, "frozen_files": frozen,
             "status": "waiting_for_qualified_fit", "learned_pilot_executed": False}

    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")

    def verify_frozen():
        if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
            raise RuntimeError("Declared postfit workflow/protocol/runner changed; no silent substitution")

    save()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            verify_frozen()
            try:
                current = json.loads((workflow / "workflow_status.json").read_text())
            except json.JSONDecodeError:
                current = None  # Only tolerate a partial publication, not a failed fit.
            if current is not None and fit_finished(current, args.updates):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Fit not complete within declared bounded wait")
            time.sleep(20)
        checkpoint, identity = ready_checkpoint(workflow, args.updates)
        if checkpoint != expected:
            raise RuntimeError("Completed fit differs from the checkpoint locked before pilot outcomes")
        validate_protocol(args.protocol.resolve(), args.oracle_audit.resolve())
        state.update(status="running_locked_pilot", checkpoint_content_sha256=identity["content_tree_sha256"])
        save()
        python = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
        command = [str(python), "-m", "VLA.run_evaluation_protocol", "--model", "smolvla",
                   "--protocol", str(args.protocol.resolve()), "--oracle-audit", str(args.oracle_audit.resolve()),
                   "--checkpoint", str(checkpoint), "--split", "development", "--replan-steps", "1",
                   "--run-tag", args.run_tag, "--output-root", str(args.output / "evaluations")]
        state["command"] = command
        save()
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}
        with (args.output / "pilot.log").open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        verify_frozen()
        verify_fingerprint(checkpoint, full=True)
        if result.returncode:
            raise RuntimeError("Pilot integration failed; preserve results/log, do not reroll failures")
        state.update(status="completed_locked_pilot", learned_pilot_executed=True)
        save()
    except Exception as exc:
        state.update(status="stopped_no_retry", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
