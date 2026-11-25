"""Finite CPU-only audit/replay continuation; NEVER starts a fine-tune.

Waits for the whole predeclared paired collection, rejects partial/failed data,
then ends at a separate actual-image-review barrier. Existing sources/checkpoints
and outer validation are never edited or used to collect corrective labels.
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
from VLA.common.source_payload_provenance import verify_source_payload_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def validate_completed(source):
    source = Path(source)
    plan = json.loads((source / "collection_plan.json").read_text())
    rows = json.loads((source / "index.json").read_text())
    audit = json.loads((source / "collection_audit.json").read_text())
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    if (not audit.get("passed") or audit.get("index_sha256") != sha256_file(source / "index.json")
            or manifest.get("candidate_only") is not True
            or len(rows) != 2 * len(plan["selected_source_files"])
            or any(not row.get("success") or row.get("safety_violation") is not None for row in rows)
            or plan.get("validation_used_for_collection") is not False or plan.get("final_test_used") is not False
            or plan.get("gentle_near_touch") is not True):
        raise ValueError("Paired source is incomplete, failed, leaking or uses the wrong touch protocol")
    counts = dict(Counter(row["task"]["name"] for row in rows))
    if set(counts) != {"touch", "pick", "move", "pick_place"} or counts != manifest["declared_task_counts"]:
        raise ValueError("Missing four-task predeclared collection")
    verify_source_payload_fingerprint(audit.get("source_payload_fingerprint"), source, rows)
    if plan.get("recovery_start_kind"):
        required = {"initial_actual_closure_at_least_95pct", "first_expert_label_reopens",
                    "no_false_grasp_on_first_action", "verified_carry_at_task_end", "no_safety_abort"}
        if (plan["recovery_start_kind"] != "closed_empty_near_object_missed_close"
                or manifest.get("recovery_start_kind") != plan["recovery_start_kind"]
                or audit.get("near_object_recovery_checks_passed") is not True
                or plan.get("startup_actions_are_training_labels") is not False
                or plan.get("source_objects_modified") is not False):
            raise ValueError("Incorrect or unverified recovery candidate protocol")
        for row in rows:
            checks = row.get("near_object_recovery_checks", {})
            if (set(checks) != required or any(checks[key] is not True for key in required)
                    or not row.get("recovery_start_actions_not_training_labels")):
                raise ValueError("Incomplete physical closed-empty/reopening/carry recovery evidence")
    return counts


def wait_for_complete(source, seconds, *, clock=time.monotonic, sleep=time.sleep):
    """Bounded wait, tolerant ONLY of an in-progress JSON publication race.

    A semantic collection failure is never retried/ignored. An indefinitely
    malformed file times out and cannot qualify data or launch training.
    """
    source = Path(source)
    deadline = clock() + seconds
    while True:
        if clock() >= deadline:
            raise TimeoutError("Full candidate collection did not complete within bounded wait")
        try:
            state_file = source / "collection_state.json"
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            if state.get("status") in {"stopped_error_NOT_training", "completed_candidate_audit_failed"}:
                raise ValueError("Failed collection retained: " + str(state.get("error")))
            index = source / "index.json"
            if index.exists():
                rows = json.loads(index.read_text())
                if any(not row.get("success") or row.get("safety_violation") is not None for row in rows):
                    raise ValueError("Failed candidate retained; no partial-source audit or fit")
            if (source / "collection_audit.json").exists():
                return validate_completed(source)
        except (FileNotFoundError, json.JSONDecodeError):
            # Collector writes may briefly be absent/partial. Only these I/O
            # publication conditions are tolerated, never failed audit values.
            pass
        sleep(min(10., max(0., deadline - clock())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smol-image-config", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=10800)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 14400 or args.output.exists():
        raise ValueError("Use a fresh output and a bounded wait <=4h")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True)
    state = {"scope": "paired_candidate_CPU_audit_ONLY", "training_performed": False,
             "source": str(args.source), "status": "running", "stage": "wait_collection", "history": [], "error": None}
    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        counts = wait_for_complete(args.source, args.wait_seconds)
        state["expected_task_counts"] = counts
        state["source_index_sha256"] = sha256_file(args.source / "index.json")
        sim = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
        rdt = ROOT / "VLA/.venvs/rdt/bin/python"
        commands = [
            ("source_audit", sim, "VLA.audit_finetune_data", ["--dataset", args.source, "--output", args.output / "source_audit",
             "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract"], args.output / "source_audit/audit.json"),
            ("canonical_native_replay", sim, "VLA.pi0.audit_closed_loop_replay", ["--source", args.source, "--output", args.output / "full_replay.json",
             "--mode", "both", "--native-success-mode", "strict", "--workers", "2"], args.output / "full_replay.json"),
            ("smol_native_images", sim, "VLA.audit_native_camera_tensors", ["--source", args.source, "--model", "smolvla",
             "--checkpoint", args.smol_image_config, "--output", args.output / "native_smol512"], None),
            ("rdt_native_images", rdt, "VLA.audit_native_camera_tensors", ["--source", args.source, "--model", "rdt",
             "--processor", ROOT / "VLA/checkpoints/siglip-so400m-patch14-384", "--output", args.output / "native_rdt384"], None),
        ]
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1",
                           HF_HOME=str(ROOT / "VLA/checkpoints/huggingface"))
        for name, python, module, tokens, report in commands:
            state["stage"] = name
            save()
            command = [str(python), "-m", module, *map(str, tokens)]
            with (args.output / (name + ".log")).open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=10800)
            state["history"].append({"stage": name, "exit_code": result.returncode, "command": command})
            if result.returncode:
                raise RuntimeError(name + " failed; no source skipping or automatic training")
            if report:
                document = json.loads(report.read_text())
                if document.get("passed") is not True or document.get("failures"):
                    raise RuntimeError(name + " evidence failed")
        state.update(status="awaiting_actual_native_image_review_NOT_training", stage="manual_visual_review")
        save()
    except Exception as exc:
        state.update(status="stopped_error_NOT_training", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
