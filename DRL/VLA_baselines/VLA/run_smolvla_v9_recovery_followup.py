"""Predeclared recovery-enriched V9 diagnostics after the primary pilot exits.

No training or benchmark changes. By default this prints a read-only plan.
--run freezes source/proof hashes and metadata-selected filenames BEFORE a
bounded wait, then invokes the existing learned-policy harness sequentially.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.evaluate_smolvla_v9_source import TASKS, select_episodes, verify_source_audit

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).with_name("evaluate_smolvla_v9_source.py")
SCOPE = "recovery_enriched_source_scene_diagnostic_not_final_benchmark"
RECOVERY_COUNTS = {
    "train": {"touch": 2, "pick": 2, "move": 0, "pick_place": 2},
    "heldout": {"touch": 0, "pick": 0, "move": 2, "pick_place": 0},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--validation-audit", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, default=ROOT / "VLA/third_party/lerobot/.venv/bin/python")
    parser.add_argument("--driver-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path,
                        help="Retry a failed declaration without changing its scenes, seed, harness or checkpoint")
    parser.add_argument("--wait-timeout", type=float, default=7200,
                        help="Maximum primary-pilot wait in seconds, bounded to at most one day")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("source", "validation", "checkpoint", "driver_status", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Preserve virtualenv executable symlinks: resolving loses pyvenv.cfg.
    args.model_python = args.model_python.expanduser().absolute()
    args.source_audit = (args.source_audit or args.source / "audit.json").expanduser().resolve()
    args.validation_audit = (args.validation_audit or args.validation / "audit.json").expanduser().resolve()
    if args.previous_plan is not None:
        args.previous_plan = args.previous_plan.expanduser().resolve()
    if not math.isfinite(args.wait_timeout) or not 0 < args.wait_timeout <= 86400:
        raise ValueError("wait-timeout must be finite and in (0, 86400] seconds")
    if args.source == args.validation:
        raise ValueError("TRAIN and VALIDATION must be separate frozen sources")
    return args


def is_recovery(row):
    """Inspect reset metadata only, never neural or teacher success outcomes."""
    startup = row.get("recovery_start_actions_not_training_labels", [])
    if not isinstance(startup, list):
        raise ValueError("Recovery startup metadata must be a list")
    return bool(startup)


def select_recovery_enriched(rows, split):
    if split not in RECOVERY_COUNTS:
        raise ValueError("Unknown diagnostic split")
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source episode filename")
    if {row["task"]["name"] for row in rows} != set(TASKS):
        raise ValueError("Each source must contain exactly the four task primitives")
    recovery = [row for row in rows if is_recovery(row)]
    counts = Counter(row["task"]["name"] for row in recovery)
    if any(counts[task] != RECOVERY_COUNTS[split][task] for task in TASKS):
        raise ValueError("Frozen recovery metadata counts do not match this follow-up protocol")
    # Default deterministic layout-spread selection supplies ordinary fillers.
    # Recovery variants intentionally remain together even on one layout.
    ordinary = select_episodes([row for row in rows if not is_recovery(row)], per_task=2)
    filenames = []
    for task in TASKS:
        task_recovery = [row for row in recovery if row["task"]["name"] == task]
        task_ordinary = [row for row in ordinary if row["task"]["name"] == task]
        filenames.extend(row["file"] for row in task_recovery)
        filenames.extend(row["file"] for row in task_ordinary[:2 - len(task_recovery)])
    return select_episodes(rows, per_task=2, filenames=filenames)


def selection_metadata(selected):
    from VLA.audit_training_readiness import physical_scene_key

    entries = [{"file": row["file"], "task": row["task"]["name"],
                "group": "recovery" if is_recovery(row) else "ordinary",
                "recovery_startup_metadata_present": is_recovery(row),
                "physical_scene_group": physical_scene_key(row["task"])} for row in selected]
    return {"episodes": entries, "episode_files": [row["file"] for row in entries],
            "trials": len(entries), "task_counts": dict(Counter(row["task"] for row in entries)),
            "recovery_trials": sum(row["group"] == "recovery" for row in entries),
            "physical_layouts": len({row["physical_scene_group"] for row in entries}),
            "recovery_physical_layouts": len({row["physical_scene_group"] for row in entries if row["group"] == "recovery"}),
            "selection_uses_neural_outcomes": False,
            "same_layout_recovery_variants_are_independent_layouts": False,
            "startup_actions_in_predeclaration": False}


def verify_source_inputs(args):
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout

    train_audit, train = verify_source_audit(args.source, args.source_audit)
    val_audit, val = verify_source_audit(args.validation, args.validation_audit)
    if train.get("split") != "train" or val.get("split") not in {"val", "validation"}:
        raise ValueError("Only frozen TRAIN/VALIDATION sources are supported")
    if train.get("final_test_used") or val.get("final_test_used"):
        raise ValueError("Final-test sources are outside this diagnostic")
    if train.get("candidate_only") is not False or train.get("training_qualification", {}).get("scope") != "bounded_pipeline_pilot":
        raise ValueError("Requires the actual qualified pilot TRAIN source")
    validate_holdout(args.source, args.validation)
    return {"train_payload_tree_sha256": train_audit["source_payload_fingerprint"]["payload_tree_sha256"],
            "validation_payload_tree_sha256": val_audit["source_payload_fingerprint"]["payload_tree_sha256"],
            "observation_profile_sha256": train["observation_profile_sha256"]}


def prepare_plan(args):
    source_contract = verify_source_inputs(args)
    sources = {"train": (args.source, args.source_audit), "heldout": (args.validation, args.validation_audit)}
    selections, stages, frozen = {}, [], []
    for split, (source, audit) in sources.items():
        rows = json.loads((source / "index.json").read_text())
        selections[split] = selection_metadata(select_recovery_enriched(rows, split))
        for path in (source / "index.json", source / "dataset_manifest.json", audit):
            frozen.append({"path": str(path), "sha256": sha256_file(path)})
        command = [str(args.model_python), "-m", "VLA.evaluate_smolvla_v9_source",
                   "--source", str(source), "--audit-report", str(audit), "--checkpoint", str(args.checkpoint),
                   "--split-mode", "train_diagnostic" if split == "train" else "heldout",
                   "--training-source", str(args.source), "--training-audit-report", str(args.source_audit),
                   "--episodes-per-task", "2", "--seed", "20261009", "--output", str(args.output / (split + "_closedloop")),
                   "--episode-files", *selections[split]["episode_files"]]
        stages.append({"name": split + "_closedloop", "split": split, "command": command,
                       "output": str(args.output / (split + "_closedloop")), "gpu": True})
    for path in (Path(__file__), HARNESS):
        frozen.append({"path": str(path), "sha256": sha256_file(path)})
    plan = {"scope": SCOPE, "source": str(args.source), "validation": str(args.validation),
            "source_audit": str(args.source_audit), "validation_audit": str(args.validation_audit),
            "checkpoint": str(args.checkpoint), "model_python": str(args.model_python),
            "driver_status": str(args.driver_status), "wait_timeout_seconds": args.wait_timeout,
            "source_contract": source_contract, "frozen_files": frozen, "selections": selections,
            "stages": stages, "training_normalization_source": str(args.source),
            "selection_uses_neural_outcomes": False, "privileged_information_in_controls": False,
            "oracle_fallback": False, "final_test_used": False, "training_performed": False,
            "primary_diagnostic_is_not_a_success_gate": True,
            "subset_success_is_not_a_benchmark_success_rate": True}
    if args.previous_plan is not None:
        inherit_predeclared_selection(plan, args.previous_plan)
    return plan


def inherit_predeclared_selection(plan, previous_path):
    """Rebind driver code only; never bypass changed source/harness/selection.

    The original failed declaration is retained and hash-bound. A fresh output
    records this as a retry, not a new subset chosen after observed outcomes.
    """
    previous_path = Path(previous_path).resolve()
    previous = json.loads(previous_path.read_text())
    keys = ("scope", "source", "validation", "source_audit", "validation_audit",
            "checkpoint", "model_python", "driver_status", "source_contract",
            "selections", "training_normalization_source", "selection_uses_neural_outcomes",
            "privileged_information_in_controls", "oracle_fallback", "final_test_used")
    if any(key not in previous or previous[key] != plan[key] for key in keys):
        raise ValueError("Retry differs from the original predeclared diagnostic contract")
    def comparable_stages(stages):
        result = []
        for stage in stages:
            command = list(stage["command"])
            position = command.index("--output")
            command[position + 1] = "<fresh-output>"
            result.append({"name": stage["name"], "split": stage["split"],
                           "gpu": stage["gpu"], "command": command})
        return result
    if comparable_stages(previous["stages"]) != comparable_stages(plan["stages"]):
        raise ValueError("Retry changes predeclared inference arguments or seed")
    driver = str(Path(__file__).resolve())
    old_hashes = {row["path"]: row["sha256"] for row in previous["frozen_files"]}
    new_hashes = {row["path"]: row["sha256"] for row in plan["frozen_files"]}
    if set(old_hashes) != set(new_hashes):
        raise ValueError("Retry changes the frozen dependency list")
    changed = [path for path in old_hashes if old_hashes[path] != new_hashes[path]]
    if any(path != driver for path in changed):
        raise ValueError("Retry cannot rebind a changed source, proof or harness")
    plan["previous_declaration"] = {"path": str(previous_path), "sha256": sha256_file(previous_path),
        "selection_and_inference_arguments_unchanged": True,
        "driver_rebound": driver in changed,
        "previous_driver_sha256": old_hashes[driver], "current_driver_sha256": new_hashes[driver]}
    plan["frozen_files"].append({"path": str(previous_path), "sha256": sha256_file(previous_path)})


def assert_frozen_inputs(args, plan):
    for record in plan["frozen_files"]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError("Predeclared input or harness changed: " + record["path"])
    if verify_source_inputs(args) != plan["source_contract"]:
        raise RuntimeError("Predeclared source contract or indexed payloads changed")


def validate_primary_completion(args, plan, status):
    if status.get("status") != "completed_diagnostic_only":
        raise ValueError("Primary pilot is not completed_diagnostic_only")
    for key in ("source", "validation"):
        if not status.get(key) or Path(status[key]).resolve() != Path(plan[key]):
            raise ValueError("Primary pilot source binding differs from the follow-up: " + key)
    checkpoints = []
    for stage in status.get("stages", []):
        command = stage.get("command", [])
        if stage.get("name") in {"train_closedloop", "heldout_closedloop"} and "--checkpoint" in command:
            checkpoints.append(Path(command[command.index("--checkpoint") + 1]).resolve())
    if len(checkpoints) != 2 or any(path != args.checkpoint for path in checkpoints):
        raise ValueError("Follow-up checkpoint is not the primary pilot's final diagnostic checkpoint")
    harness_hash = next(row["sha256"] for row in plan["frozen_files"] if Path(row["path"]) == HARNESS)
    if status.get("harness_sha256") != harness_hash:
        raise ValueError("Primary pilot and follow-up harness provenance differ")


def wait_for_primary(path, timeout, *, clock=time.monotonic, sleep=time.sleep, reader=None, on_poll=None):
    """Finite CPU-only wait; transient partial JSON writes do not trigger CUDA."""
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("Invalid bounded primary wait")
    if reader is None:
        reader = lambda: json.loads(Path(path).read_text())
    deadline = clock() + timeout
    while True:
        try:
            primary = reader()
        except (FileNotFoundError, json.JSONDecodeError):
            primary = None
        now = clock()
        if now >= deadline:
            raise TimeoutError(f"Primary pilot did not complete within {timeout:g} seconds")
        if primary is not None:
            if not isinstance(primary, dict):
                raise ValueError("Primary status must be a JSON object")
            state = primary.get("status")
            if state == "completed_diagnostic_only":
                return primary
            if state == "failed":
                raise RuntimeError("Primary pilot failed: " + str(primary.get("error")))
            if state not in {"running", "stage_completed"}:
                raise ValueError("Unrecognized primary pilot status: " + str(state))
        if on_poll is not None:
            on_poll(primary)
        sleep(min(15.0, deadline - now))


def subgroup_summary(report, selected):
    """Summarize filenames predeclared as recovery/ordinary, never reselect."""
    expected = {row["file"]: row for row in selected["episodes"]}
    results = report.get("results", [])
    names = [row["file"] for row in results]
    if len(names) != len(set(names)) or set(names) != set(expected) or len(names) != 8:
        raise ValueError("Harness results do not match the eight predeclared episode files")
    if any(row["task"] != expected[row["file"]]["task"] or type(row["success"]) is not bool for row in results):
        raise ValueError("Harness task/result labels differ from predeclaration")
    summary = {}
    for group in ("recovery", "ordinary"):
        rows = [row for row in results if expected[row["file"]]["group"] == group]
        summary[group] = {"trials": len(rows), "successes": sum(row["success"] for row in rows),
                          "diagnostic_success_fraction": None if not rows else sum(row["success"] for row in rows) / len(rows),
                          "physical_layouts": len({expected[row["file"]]["physical_scene_group"] for row in rows}),
                          "safety_aborts": sum(bool(row.get("safety_abort")) for row in rows),
                          "correct_target_grasp_seen": sum(bool(row.get("correct_target_grasp_seen")) for row in rows),
                          "wrong_target_grasp_seen": sum(bool(row.get("wrong_target_grasp_seen")) for row in rows),
                          "episode_files": [row["file"] for row in rows]}
    return summary


def validate_result_report(args, stage, report):
    expected_source = args.source if stage["split"] == "train" else args.validation
    for key, expected in (("source", expected_source), ("training_normalization_source", args.source), ("checkpoint", args.checkpoint)):
        if not report.get(key) or Path(report[key]).resolve() != expected:
            raise ValueError("Harness report source/checkpoint binding changed: " + key)
    expected_mode = "train_diagnostic" if stage["split"] == "train" else "heldout"
    if report.get("split_mode") != expected_mode or report.get("trials") != 8:
        raise ValueError("Harness report split or trial count changed")
    if any(report.get(key) is not False for key in ("privileged_information_in_controls", "oracle_fallback", "final_test_used")):
        raise ValueError("Harness report lacks the nonprivileged diagnostic contract")


def run_followup(args, plan):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite follow-up output: " + str(args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "logs"
    logs.mkdir()
    plan_path = args.output / "followup_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    frozen_plan_sha256 = sha256_file(plan_path)
    history = []
    state_path = args.output / "followup_status.json"

    def status(state, stage, error=None):
        state_path.write_text(json.dumps({"scope": SCOPE, "status": state, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "predeclared_plan_sha256": frozen_plan_sha256,
            "history": history, "error": error}, indent=2) + "\n")

    stage_name = "waiting_primary"
    try:
        # The immutable declaration exists before any primary-status wait.
        status("waiting_primary", stage_name)
        primary = wait_for_primary(args.driver_status, args.wait_timeout)
        if sha256_file(plan_path) != frozen_plan_sha256:
            raise RuntimeError("Predeclared follow-up plan changed during primary wait")
        validate_primary_completion(args, plan, primary)
        assert_frozen_inputs(args, plan)
        from VLA.fingerprint_checkpoint import verify_fingerprint
        identity = verify_fingerprint(args.checkpoint, full=True)
        (args.output / "primary_completion.json").write_text(json.dumps(primary, indent=2) + "\n")
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                       "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
                       "TOKENIZERS_PARALLELISM": "false"}
        summaries = {}
        for stage in plan["stages"]:
            stage_name = stage["name"]
            if sha256_file(plan_path) != frozen_plan_sha256:
                raise RuntimeError("Predeclared follow-up plan changed before evaluation")
            assert_frozen_inputs(args, plan)
            if verify_fingerprint(args.checkpoint, full=True)["content_tree_sha256"] != identity["content_tree_sha256"]:
                raise RuntimeError("Completed checkpoint changed before follow-up")
            status("running", stage_name)
            log_path = logs / (stage_name + ".log")
            with log_path.open("x") as stream:
                completed = subprocess.run(stage["command"], cwd=ROOT, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
            history.append({"stage": stage_name, "exit_code": completed.returncode, "log": str(log_path)})
            if completed.returncode:
                raise RuntimeError(f"Follow-up stage exited {completed.returncode}: {log_path}")
            result_path = Path(stage["output"]) / "results.json"
            report = json.loads(result_path.read_text())
            validate_result_report(args, stage, report)
            summaries[stage["split"]] = {"scope": report["scope"], "report": str(result_path),
                "report_sha256": sha256_file(result_path), "subgroups": subgroup_summary(report, plan["selections"][stage["split"]])}
            status("stage_completed", stage_name)
        assert_frozen_inputs(args, plan)
        if sha256_file(plan_path) != frozen_plan_sha256:
            raise RuntimeError("Predeclared follow-up plan changed during evaluation")
        if verify_fingerprint(args.checkpoint, full=True)["content_tree_sha256"] != identity["content_tree_sha256"]:
            raise RuntimeError("Completed checkpoint changed during follow-up")
        (args.output / "results.json").write_text(json.dumps({"scope": SCOPE,
            "predeclared_plan_sha256": frozen_plan_sha256,
            "checkpoint_content_sha256": identity["content_tree_sha256"], "training_normalization_source": str(args.source),
            "selection_uses_neural_outcomes": False, "privileged_information_in_controls": False,
            "oracle_fallback": False, "final_test_used": False,
            "same_layout_recovery_variants_are_independent_layouts": False, "splits": summaries}, indent=2) + "\n")
        status("completed_diagnostic_only", "done")
    except Exception as exc:
        status("failed", stage_name, str(exc))
        raise


def main(argv=None):
    args = parse_args(argv)
    plan = prepare_plan(args)
    if not args.run:
        print(json.dumps({"run": False, **plan}, indent=2))
        return
    run_followup(args, plan)


if __name__ == "__main__":
    main()
