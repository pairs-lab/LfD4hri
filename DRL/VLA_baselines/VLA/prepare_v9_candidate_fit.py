"""Finite, fail-closed candidate audit -> qualified fit -> online evaluation.

Collection failures STOP this job; no scene filtering or automatic relaxation.
Actual exported images require a separately hash-bound visual-review receipt
before CUDA training. Source quality is never a learned-success certificate.
"""
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

ROOT = Path(__file__).resolve().parents[1]


def fit_recipe_tokens(initial_frame_probability=.30, learning_rate=1e-4, batch_size=1):
    """One declared recipe, forwarded to BOTH smoke and fit by the pilot.

    Recipe plumbing is not permission to bypass qualification or actual review.
    """
    if (not math.isfinite(initial_frame_probability) or not 0 < initial_frame_probability < 1
            or not math.isfinite(learning_rate) or not 0 < learning_rate < 1
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Invalid explicit sampling, learning rate or batch size")
    return ["--initial-frame-sampling-probability", str(initial_frame_probability),
            "--learning-rate", str(learning_rate), "--batch-size", str(batch_size)]


def collection_ready(source, expected):
    expected_counts = ({name: expected for name in ("touch", "pick", "move", "pick_place")}
                       if type(expected) is int else expected)
    if (not isinstance(expected_counts, dict) or set(expected_counts) != {"touch", "pick", "move", "pick_place"}
            or any(type(value) is not int or value < 1 for value in expected_counts.values())):
        raise ValueError("Predeclare positive counts for all four tasks")
    path = Path(source)/"collection_state.json"
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state["status"].startswith("stopped"):
        raise RuntimeError("Collector preserved a failure/exception; diagnose exact scene before fitting")
    if state["status"] == "in_progress":
        return False
    if state["status"] != "completed_candidate_not_qualified":
        raise RuntimeError("Unknown collection terminal status")
    rows = json.loads((Path(source)/"index.json").read_text())
    counts = Counter(row["task"]["name"] for row in rows)
    if (counts != expected_counts
            or len(rows) != state["completed_episodes"]
            or sha256_file(Path(source)/"index.json") != state["index_sha256"]
            or any(not row["success"] or row.get("safety_violation") for row in rows)):
        raise RuntimeError("Incomplete, changed or unsafe collection")
    return True


def require_visual_receipt(path, source, native, source_audit):
    receipt = json.loads(Path(path).read_text())
    required = {"source_index_sha256": sha256_file(Path(source)/"index.json"),
                "native_report_sha256": sha256_file(Path(native)/"audit.json"),
                "source_audit_sha256": sha256_file(source_audit)}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Visual review is not bound to this exact audited candidate")
    if (receipt.get("reviewer") != "codex_actual_image_inspection"
            or set(receipt.get("tasks_reviewed", [])) != {"touch", "pick", "move", "pick_place"}
            or not receipt.get("acceptable_for_bounded_fit") or not receipt.get("images")):
        raise ValueError("Missing actual four-task visual review; no training")
    mandatory = {str((Path(native)/"native_inputs_contact_sheet.png").resolve()),
                 str((Path(source_audit).parent/"camera_contact_sheet.png").resolve())}
    inspected = set()
    for item in receipt["images"]:
        image = Path(item["path"]).resolve()
        if sha256_file(image) != item["sha256"]:
            raise ValueError("Reviewed image changed")
        inspected.add(str(image))
    if not mandatory <= inspected:
        raise ValueError("Review omits actual native or source four-task contact sheet")


def additional_collection_ready(source, expected_per_task):
    """Do not merge a live recovery prefix; legacy complete pairs have no state."""
    source = Path(source)
    state_path = source / "collection_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state["status"].startswith("stopped"):
            raise RuntimeError("Additional collector preserved a failure; no merge or fit")
        if state["status"] == "in_progress":
            return False
        if state["status"] not in ("completed_candidate_not_qualified", "completed_candidate_audit_passed"):
            raise RuntimeError("Unknown additional collection terminal status")
    else:
        state = None
    index_path, manifest_path = source / "index.json", source / "dataset_manifest.json"
    if not index_path.is_file() or not manifest_path.is_file():
        return False
    index, manifest = json.loads(index_path.read_text()), json.loads(manifest_path.read_text())
    counts = Counter(row["task"]["name"] for row in index)
    expected = {name: expected_per_task for name in ("touch", "pick", "move", "pick_place")}
    digest = sha256_file(index_path)
    if (counts != expected or manifest.get("episodes") != len(index)
            or manifest.get("index_sha256") != digest
            or any(not row.get("success") or row.get("safety_violation") for row in index)
            or (state is not None and (state.get("completed_episodes") != len(index)
                or state.get("index_sha256", digest) != digest))):
        raise RuntimeError("Additional source is incomplete, changed, or unsafe")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--additional-source", type=Path, action="append", default=[])
    parser.add_argument("--additional-expected-per-task", type=int, default=38)
    parser.add_argument("--recovery-reset-observability", choices=("front", "dual_view"), default="front")
    parser.add_argument("--comparison-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, help="Separate VLA/model/experiment/test folder for checkpoints and rollout videos")
    parser.add_argument("--preview-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-per-task", type=int, default=160)
    parser.add_argument("--expected-task-counts", type=json.loads,
                        help="Exact predeclared counts for a nonuniform repaired candidate; never infer/filter completed successes")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--paired-grounding-probability", type=float, default=0.)
    parser.add_argument("--initial-frame-sampling-probability", type=float, default=.30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.additional_expected_per_task < 1:
        raise ValueError("Predeclare positive additional per-task counts")
    recipe = fit_recipe_tokens(args.initial_frame_sampling_probability, args.learning_rate, args.batch_size)
    if args.output.exists() or not 2 <= args.updates <= 20000 or not 1 <= args.wait_seconds <= 86400 or not 0 <= args.paired_grounding_probability < 1:
        raise ValueError("Use NEW output and a finite qualified pilot budget")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    model_output = (args.model_output or output/"smolvla/test1").resolve()
    if model_output.exists():
        raise FileExistsError("Do not overwrite an existing model experiment")
    status_path = output/"workflow_status.json"
    logs = output/"logs"
    logs.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                   "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT/"VLA/checkpoints/huggingface"),
                   "TOKENIZERS_PARALLELISM": "false"}
    python = str(ROOT/"VLA/third_party/lerobot/.venv/bin/python")
    history = []
    contract = {"candidate": str(args.candidate.resolve()), "updates": args.updates,
                "additional_sources": [str(p.resolve()) for p in args.additional_source],
                "additional_expected_per_task": args.additional_expected_per_task,
                "recovery_reset_observability": args.recovery_reset_observability,
                "source_camera_gate_scope": "dual-view SmolVLA only; NOT front-only OpenVLA qualification",
                "expected_candidate_counts": args.expected_task_counts or args.expected_per_task,
                "paired_grounding_probability": args.paired_grounding_probability,
                "initial_frame_sampling_probability": args.initial_frame_sampling_probability,
                "effective_initial_frame_sampling_probability": args.paired_grounding_probability +
                    (1-args.paired_grounding_probability)*args.initial_frame_sampling_probability,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "per_wait_deadline_seconds": args.wait_seconds,
                "model_output": str(model_output),
                "scope": "bounded_fit_then_diagnostic_rollouts_NOT_final_benchmark",
                "data_sufficiency_established": False, "run_training_requested": args.run_training,
                "final_test_used": False, "workflow_sha256": sha256_file(Path(__file__))}
    (output/"workflow_plan.json").write_text(json.dumps(contract, indent=2)+"\n")
    def status(state, stage, error=None):
        status_path.write_text(json.dumps({**contract, "status": state, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "history": history, "error": error}, indent=2)+"\n")
    def stage(name, module, tokens, report=None, pass_key="passed"):
        if sha256_file(Path(__file__)) != contract["workflow_sha256"]:
            raise RuntimeError("Workflow implementation changed during execution")
        status("running", name)
        command = [python, "-m", module, *map(str, tokens)]
        with (logs/(name+".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"stage": name, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {logs/(name+'.log')}")
        if report is not None and not json.loads(Path(report).read_text()).get(pass_key):
            raise RuntimeError(f"{name} report failed: {report}")
        status("stage_completed", name)
    def source_audit(name, source):
        rows = json.loads((source/"index.json").read_text())
        counts = Counter(row["task"]["name"] for row in rows)
        report = output/name/"audit.json"
        stage(name, "VLA.audit_finetune_data", ["--dataset", source, "--output", report.parent,
            "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract",
            "--recovery-reset-observability", args.recovery_reset_observability], report)
        return report
    try:
        deadline = time.monotonic()+args.wait_seconds
        status("waiting_for_complete_collection", "collection")
        while not collection_ready(args.candidate, args.expected_task_counts or args.expected_per_task):
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded collection wait expired; no training started")
            time.sleep(20)
        status("waiting_for_complete_additional_collection", "additional_collection")
        while not all(additional_collection_ready(path, args.additional_expected_per_task) for path in args.additional_source):
            if time.monotonic() >= deadline:
                raise TimeoutError("Additional collection wait expired; no merge/fit started")
            time.sleep(20)
        combined = output/"combined_candidate"
        tokens = ["--source", args.candidate]
        for extra in args.additional_source:
            tokens.extend(["--source", extra])
        stage("merge", "VLA.merge_v9_train_candidates", [*tokens, "--validation", args.comparison_validation, "--output", combined])
        audit = source_audit("combined_source_audit", combined)
        # Reject known sensor outages BEFORE paying for exhaustive physics
        # replay or GPU fitting. The existing qualification gate remains and
        # rechecks the exact complete source; no visibility rule is relaxed.
        visibility = output/"combined_transport_visibility.json"
        stage("transport_visibility", "VLA.audit_transport_visibility", ["--source", combined,
            "--output", visibility, "--max-transient-frames", "1"], visibility)
        replay = output/"combined_full_physics_replay.json"
        stage("full_physics_replay", "VLA.pi0.audit_closed_loop_replay", ["--source", combined,
            "--output", replay, "--mode", "both", "--native-success-mode", "strict", "--workers", "2"], replay)
        native = output/"native_smol512"
        stage("native_camera", "VLA.audit_native_camera_tensors", ["--source", combined,
            "--model", "smolvla", "--checkpoint", args.preview_checkpoint, "--output", native])
        readiness = output/"readiness/readiness.json"
        stage("readiness", "VLA.audit_training_readiness", ["--source", combined, "--source-audit", audit,
            "--output", readiness.parent, "--preserve-training-vocabulary", "--minimum-train-repetitions",
            "3" if args.updates > 1000 else "1"], readiness, "source_integrity_passed")
        train, validation = output/"train_candidate", output/"validation_candidate"
        for split, path in (("train", train), ("validation", validation)):
            stage("materialize_"+split, "VLA.materialize_training_split", ["--readiness", readiness, "--split", split, "--output", path])
        train_audit = source_audit("train_candidate_audit", train)
        validation_audit = source_audit("validation_audit", validation)
        # Explicit human-independent agent review barrier. Root must actually
        # open these exported images and record their exact bytes before fit.
        receipt = output/"visual_review_receipt.json"
        (output/"visual_review_requirements.json").write_text(json.dumps({"source": str(combined),
            "source_index_sha256": sha256_file(combined/"index.json"),
            "native_report_sha256": sha256_file(native/"audit.json"),
            "source_audit_sha256": sha256_file(audit),
            "required_images": [str(native/"native_inputs_contact_sheet.png"), str(audit.parent/"camera_contact_sheet.png")],
            "receipt": str(receipt)}, indent=2)+"\n")
        if not args.run_training:
            status("awaiting_visual_review_no_training", "review")
            return
        status("awaiting_actual_visual_review", "review")
        deadline = time.monotonic()+args.wait_seconds
        while not receipt.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Visual review was not supplied; no training started")
            time.sleep(20)
        require_visual_receipt(receipt, combined, native, audit)
        qualified = output/"qualified_train"
        qualification_tokens = ["--source", train, "--source-audit", train_audit,
            "--native-replay-report", replay, "--output", qualified, "--max-updates", args.updates]
        scope = "bounded_pipeline_pilot"
        module = "VLA.qualify_bounded_training_pilot"
        if args.updates > 1000:
            scope = "bounded_refinement_experiment"
            module = "VLA.qualify_bounded_refinement"
            qualification_tokens.extend(["--visual-review-receipt", receipt, "--native-camera-dir", native,
                "--combined-source", combined, "--combined-source-audit", audit])
        stage("qualify", module, qualification_tokens)
        qualified_audit = source_audit("qualified_train_audit", qualified)
        fit_command = ["--source", qualified, "--validation", validation,
            "--dataset-root", output/"lerobot_dataset", "--audit-root", output/"fit_audit",
            "--source-audit", qualified_audit, "--validation-audit", validation_audit,
            "--output", model_output, "--updates", args.updates, "--experiment-scope", scope, *recipe]
        if args.paired_grounding_probability:
            fit_command.extend(["--paired-grounding-probability", str(args.paired_grounding_probability)])
        stage("fit_plan", "VLA.run_smolvla_v9_pilot", fit_command)
        stage("fit_and_closedloop", "VLA.run_smolvla_v9_pilot", [*fit_command, "--run"])
        status("completed_diagnostic_only", "done")
    except Exception as exc:
        status("stopped_no_silent_retry", "failure", str(exc))
        raise


if __name__ == "__main__":
    main()
