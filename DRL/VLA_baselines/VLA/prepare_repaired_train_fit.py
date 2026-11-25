"""Finite CPU preparation of ALL repaired frozen TRAIN, never outer VAL labels.

The existing qualification gates are unchanged. A nested development split
uses only the original TRAIN population; outer validation stays untouched.
Actual final-source visual inspection and model-specific conversion/smoke
remain mandatory. This program never starts a GPU optimizer.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

from VLA.common.provenance import sha256_file
from VLA.frozen_reset_rounding import compare_frozen_reset

ROOT = Path(__file__).resolve().parents[1]


def repaired_collection_ready(candidate, original_train, outer_validation):
    candidate, original_train, outer_validation = map(Path, (candidate, original_train, outer_validation))
    state_path = candidate / "collection_state.json"
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    if state["status"].startswith("stopped"):
        raise ValueError("Corrective collection preserved a failure; no scene filtering or training")
    if state["status"] == "in_progress":
        return False
    if state["status"] != "completed_candidate_not_qualified":
        raise ValueError("Unknown corrective collection terminal status")
    plan = json.loads((candidate / "repair_plan.json").read_text())
    if (plan.get("validation_labels_used") is not False or plan.get("final_test_used") is not False
            or Path(plan["train_source"]).resolve() != original_train.resolve()
            or Path(plan["outer_validation"]).resolve() != outer_validation.resolve()
            or plan["train_index_sha256"] != sha256_file(original_train / "index.json")
            or plan["train_manifest_sha256"] != sha256_file(original_train / "dataset_manifest.json")
            or plan["outer_validation_index_sha256"] != sha256_file(outer_validation / "index.json")):
        raise ValueError("Changed/non-TRAIN correction provenance")
    original_manifest = json.loads((original_train / "dataset_manifest.json").read_text())
    manifest = json.loads((candidate / "dataset_manifest.json").read_text())
    if (original_manifest.get("split") != "train" or manifest.get("split") is not None
            or manifest["observation_profile_sha256"] != original_manifest["observation_profile_sha256"]
            or manifest["environment_runtime_fingerprint"] != original_manifest["environment_runtime_fingerprint"]):
        raise ValueError("Original TRAIN split or frozen runtime changed")
    old_rows = json.loads((original_train / "index.json").read_text())
    rows = json.loads((candidate / "index.json").read_text())
    outer = json.loads((outer_validation / "index.json").read_text())
    old_by_file = {r["file"]: r for r in old_rows}
    if (len(old_by_file) != len(old_rows) or len({r["file"] for r in rows}) != len(rows)
            or set(old_by_file) != {r["file"] for r in rows}
            or len(rows) != state["planned_episodes"] or len(rows) != state["completed_episodes"]
            or state["index_sha256"] != sha256_file(candidate / "index.json")
            or Counter(r["task"]["name"] for r in rows) != plan["planned_counts"]
            or {r["scene_group"] for r in rows} & {r["scene_group"] for r in outer}):
        raise ValueError("Missing, duplicated or leaking corrective cases")
    for row in rows:
        old = old_by_file[row["file"]]
        if (row["task"] != old["task"] or row["scene_group"] != old["scene_group"]
                or not row["success"] or row.get("safety_violation")):
            raise ValueError("Changed scene/task, failed or unsafe corrective case")
        if row["task"]["name"] == "touch":
            if (row.get("expert_touch_terminal_semantics") != "original_strict_AND_surface_gap_le_0.2mm"
                    or not 0 <= row["expert_terminal_surface_gap_m"] <= .0002
                    or row["expert_terminal_max_normal_force_n"] > 5):
                raise ValueError("Not ALL touch cases use the audited near-surface protocol")
            # The 19 seed preflights record reset_exact in their report, not
            # their episode metadata. Independently verify EVERY reset here;
            # do not invent a flag or weaken the requirement for those rows.
            with np.load(candidate / row["file"], allow_pickle=False) as new, np.load(original_train / row["file"], allow_pickle=False) as previous:
                compare_frozen_reset(new, previous)
        elif (row.get("non_touch_payload_unchanged") is not True
                or sha256_file(candidate / row["file"]) != sha256_file(original_train / row["file"])):
            raise ValueError("Non-touch payload changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "original-train", "outer-validation", "output", "preview-checkpoint", "model-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=10800)
    parser.add_argument("--updates", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.model_output.exists() or not 1 <= args.wait_seconds <= 14400 or not 1001 <= args.updates <= 20000:
        raise ValueError("Fresh outputs and finite preparation/refinement budgets required")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    logs = output / "logs"
    logs.mkdir()
    source = args.candidate.resolve()
    outer = args.outer_validation.resolve()
    python = str(ROOT / "VLA/third_party/lerobot/.venv/bin/python")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface")}
    references = {str((p / name).resolve()): sha256_file(p / name)
        for p in (args.original_train, outer) for name in ("index.json", "dataset_manifest.json")}
    contract = {"candidate": str(source), "original_train": str(args.original_train.resolve()),
        "outer_validation": str(outer), "frozen_references": references, "updates": args.updates,
        "nested_split": "deterministic grouped 80/20 within ORIGINAL TRAIN only; actual fit budget is the inner TRAIN count",
        "scope": "bounded_refinement_preparation_NOT_final_benchmark", "data_sufficiency_established": False,
        "gpu_optimizer_started": False, "final_test_used": False, "model_output": str(args.model_output.resolve()),
        "workflow_sha256": sha256_file(Path(__file__))}
    (output / "workflow_plan.json").write_text(json.dumps(contract, indent=2) + "\n")
    history = []
    def status(state, stage, error=None):
        (output / "workflow_status.json").write_text(json.dumps({**contract, "status": state, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "history": history, "error": error}, indent=2) + "\n")
    def unchanged():
        if (sha256_file(Path(__file__)) != contract["workflow_sha256"]
                or any(sha256_file(Path(p)) != digest for p, digest in references.items())):
            raise RuntimeError("Workflow or frozen TRAIN/outer-validation references changed")
    def stage(name, module, tokens, report=None, pass_key="passed"):
        unchanged()
        status("running", name)
        command = [python, "-m", module, *map(str, tokens)]
        with (logs / (name + ".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=10800)
        history.append({"stage": name, "command": command, "exit_code": result.returncode})
        if result.returncode or (report and not json.loads(Path(report).read_text()).get(pass_key)):
            raise RuntimeError(f"{name} failed: inspect {logs / (name + '.log')}; no silent retry")
        unchanged()
        status("stage_completed", name)
    def source_audit(name, dataset):
        counts = Counter(r["task"]["name"] for r in json.loads((dataset / "index.json").read_text()))
        report = output / name / "audit.json"
        stage(name, "VLA.audit_finetune_data", ["--dataset", dataset, "--output", report.parent,
            "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract"], report)
        return report
    try:
        deadline = time.monotonic() + args.wait_seconds
        status("waiting_for_complete_collection", "collection")
        while not repaired_collection_ready(source, args.original_train, outer):
            unchanged()
            if time.monotonic() >= deadline:
                raise TimeoutError("Finite corrective collection wait expired; no optimizer started")
            time.sleep(20)
        audit = source_audit("all_repaired_train_audit", source)
        outer_audit = source_audit("outer_validation_audit", outer)
        native = output / "native_smol512"
        stage("native_camera", "VLA.audit_native_camera_tensors", ["--source", source, "--model", "smolvla",
            "--checkpoint", args.preview_checkpoint, "--output", native])
        stage("transport_visibility", "VLA.audit_transport_visibility", ["--source", source, "--output",
            output / "transport_visibility.json", "--max-transient-frames", "1"], output / "transport_visibility.json")
        readiness = output / "inner_readiness/readiness.json"
        stage("inner_readiness", "VLA.audit_training_readiness", ["--source", source, "--source-audit", audit,
            "--output", readiness.parent, "--preserve-training-vocabulary", "--minimum-train-repetitions", "3"], readiness, "source_integrity_passed")
        inner_train, inner_validation = output / "inner_train_candidate", output / "inner_validation_candidate"
        for split, path in (("train", inner_train), ("validation", inner_validation)):
            stage("materialize_inner_" + split, "VLA.materialize_training_split", ["--readiness", readiness, "--split", split, "--output", path])
        train_audit = source_audit("inner_train_audit", inner_train)
        source_audit("inner_validation_audit", inner_validation)
        stage("numeric_preparation", "VLA.prepare_frozen_numeric_contract", ["--train", inner_train, "--validation", outer,
            "--train-audit", train_audit, "--output", output / "inner_train_numeric_preparation.json"])
        replay = output / "all_repaired_train_full_physics_replay.json"
        stage("full_physics_replay", "VLA.pi0.audit_closed_loop_replay", ["--source", source, "--output", replay,
            "--mode", "both", "--native-success-mode", "strict", "--workers", "2"], replay)
        receipt = output / "visual_review_receipt.json"
        (output / "visual_review_requirements.json").write_text(json.dumps({"source": str(source),
            "source_index_sha256": sha256_file(source / "index.json"), "native_report_sha256": sha256_file(native / "audit.json"),
            "source_audit_sha256": sha256_file(audit), "required_images": [str(native / "native_inputs_contact_sheet.png"),
                str(audit.parent / "camera_contact_sheet.png")], "receipt": str(receipt)}, indent=2) + "\n")
        qualified = output / "qualified_inner_train"
        qualification = [python, "-m", "VLA.qualify_bounded_refinement", "--source", str(inner_train), "--source-audit", str(train_audit),
            "--native-replay-report", str(replay), "--output", str(qualified), "--max-updates", str(args.updates),
            "--visual-review-receipt", str(receipt), "--native-camera-dir", str(native), "--combined-source", str(source), "--combined-source-audit", str(audit)]
        qualified_audit = output / "qualified_train_audit/audit.json"
        fit = [python, "-m", "VLA.run_smolvla_v9_pilot", "--source", str(qualified), "--validation", str(outer),
            "--dataset-root", str(output / "lerobot_dataset"), "--audit-root", str(output / "fit_audit"),
            "--source-audit", str(qualified_audit), "--validation-audit", str(outer_audit),
            "--output", str(args.model_output.resolve()), "--updates", str(args.updates), "--experiment-scope", "bounded_refinement_experiment"]
        (output / "next_commands.json").write_text(json.dumps({"scope": "Ordered commands; NOT training authorization or a visual receipt",
            "review_required_first": str(receipt), "qualify": qualification,
            "qualified_source_audit": [python, "-m", "VLA.audit_finetune_data", "--dataset", str(qualified), "--output", str(qualified_audit.parent),
                "--expected-task-counts", json.dumps(Counter(r["task"]["name"] for r in json.loads((inner_train / "index.json").read_text()))), "--allow-stricter-frozen-reward-contract"],
            "fit_plan": fit, "fit_and_evaluate_after_all_gates": [*fit, "--run"]}, indent=2) + "\n")
        status("awaiting_actual_final_source_visual_review_no_training", "review")
    except BaseException as exc:
        status("stopped_no_silent_retry_no_training", "failure", str(exc))
        raise


if __name__ == "__main__":
    main()
