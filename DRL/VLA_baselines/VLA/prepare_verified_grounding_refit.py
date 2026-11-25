"""Finite grounding-refit continuation with unchanged qualification gates.

Reuses completed physical replays only after full payload/physics hash checks.
New grouped split, native images, actual review, conversion, statistics, two
optimizer-update smoke and deployment parity remain mandatory before fitting.
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
from VLA.common.source_payload_provenance import verify_source_payload_for_profile
from VLA.prepare_v9_candidate_fit import require_visual_receipt
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "validation", "validation-audit", "preview-checkpoint", "output", "model-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--parent-replay", type=Path, action="append", required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--grounding-probability", type=float, default=.2)
    args = parser.parse_args()
    if args.output.exists() or args.model_output.exists() or not 1001 <= args.updates <= 20000 or not 0 < args.grounding_probability < 1:
        raise ValueError("Fresh outputs and explicit bounded grounding recipe required")
    args.source = args.source.resolve()
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    logs = output/"logs"; logs.mkdir()
    python = str(ROOT/"VLA/third_party/lerobot/.venv/bin/python")
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(ROOT/"VLA/checkpoints/huggingface"), "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    frozen = {str(path.resolve()): sha256_file(path) for path in [args.source/"index.json", args.source/"dataset_manifest.json",
        args.source_audit, args.validation/"index.json", args.validation/"dataset_manifest.json", args.validation_audit,
        *args.parent_replay, Path(__file__)]}
    plan = {"scope": "verified_grounding_refit_NOT_final_benchmark", "source": str(args.source),
        "validation": str(args.validation.resolve()), "updates": args.updates, "grounding_probability": args.grounding_probability,
        "model_output": str(args.model_output.resolve()), "frozen_inputs": frozen, "final_test_used": False,
        "data_sufficiency_established": False, "new_physics_replays_executed": False,
        "replay_reuse_requires_byte_identical_payloads_and_unchanged_physics": True}
    (output/"workflow_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    history = []
    def status(state, stage, error=None):
        (output/"workflow_status.json").write_text(json.dumps({**plan, "status": state, "stage": stage,
            "error": error, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "history": history}, indent=2)+"\n")
    def unchanged():
        if any(sha256_file(path) != digest for path, digest in frozen.items()):
            raise RuntimeError("Frozen preparation/source/validation/proof changed")
    def stage(name, module, tokens, report=None, pass_key="passed"):
        unchanged(); status("running", name)
        command = [python, "-m", module, *map(str, tokens)]
        with (logs/(name+".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                timeout=86400 if name == "fit_and_evaluate" else 10800)
        history.append({"stage": name, "exit_code": result.returncode, "command": command})
        if result.returncode or (report and not json.loads(Path(report).read_text()).get(pass_key)):
            raise RuntimeError(name+" failed; no implicit retry or scene filtering")
        unchanged(); status("stage_completed", name)
    def source_audit(name, source):
        counts = Counter(row["task"]["name"] for row in json.loads((source/"index.json").read_text()))
        report = output/name/"audit.json"
        stage(name, "VLA.audit_finetune_data", ["--dataset", source, "--output", report.parent,
            "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract"], report)
        return report
    try:
        status("running", "verify_completed_source_audit")
        audit = json.loads(args.source_audit.read_text())
        if (audit.get("passed") is not True or audit.get("failures") or Path(audit["dataset"]).resolve() != args.source
                or audit["index_sha256"] != sha256_file(args.source/"index.json")
                or audit["manifest_sha256"] != sha256_file(args.source/"dataset_manifest.json")):
            raise ValueError("Completed source camera audit is missing or stale")
        verify_source_payload_for_profile(audit, args.source)
        validate_holdout(args.source, args.validation.resolve())
        replay = output/"verified_parent_physics_replays.json"
        tokens = ["--source", args.source, "--output", replay]
        for proof in args.parent_replay:
            tokens.extend(["--parent-replay", proof])
        stage("verified_replay_reuse", "VLA.compose_verified_physics_replays", tokens, replay)
        native = output/"native_smol512"
        stage("native_camera", "VLA.audit_native_camera_tensors", ["--source", args.source,
            "--model", "smolvla", "--checkpoint", args.preview_checkpoint, "--output", native])
        readiness = output/"readiness/readiness.json"
        stage("grouped_readiness", "VLA.audit_training_readiness", ["--source", args.source, "--source-audit", args.source_audit,
            "--output", readiness.parent, "--preserve-training-vocabulary", "--minimum-train-repetitions", "3"], readiness, "source_integrity_passed")
        train, inner_val = output/"train_candidate", output/"inner_validation_candidate"
        for split, path in (("train", train), ("validation", inner_val)):
            stage("materialize_"+split, "VLA.materialize_training_split", ["--readiness", readiness, "--split", split, "--output", path])
        train_audit = source_audit("train_candidate_audit", train)
        receipt = output/"visual_review_receipt.json"
        (output/"visual_review_requirements.json").write_text(json.dumps({"source": str(args.source),
            "source_index_sha256": sha256_file(args.source/"index.json"), "native_report_sha256": sha256_file(native/"audit.json"),
            "source_audit_sha256": sha256_file(args.source_audit), "required_images": [str(native/"native_inputs_contact_sheet.png"),
                str(args.source_audit.parent/"camera_contact_sheet.png")], "receipt": str(receipt)}, indent=2)+"\n")
        status("awaiting_actual_native_image_review_NO_optimizer", "review")
        deadline = time.monotonic()+7200
        while not receipt.exists():
            unchanged()
            if time.monotonic() >= deadline:
                raise TimeoutError("Actual native-image review not supplied within finite2h deadline")
            time.sleep(10)
        require_visual_receipt(receipt, args.source, native, args.source_audit)
        qualified = output/"qualified_train"
        stage("qualify", "VLA.qualify_bounded_refinement", ["--source", train, "--source-audit", train_audit,
            "--native-replay-report", replay, "--output", qualified, "--max-updates", args.updates,
            "--visual-review-receipt", receipt, "--native-camera-dir", native, "--combined-source", args.source,
            "--combined-source-audit", args.source_audit])
        qualified_audit = source_audit("qualified_train_audit", qualified)
        stage("fit_and_evaluate", "VLA.run_smolvla_v9_pilot", ["--source", qualified, "--validation", args.validation,
            "--dataset-root", output/"lerobot_dataset", "--audit-root", output/"fit_audit", "--output", args.model_output,
            "--source-audit", qualified_audit, "--validation-audit", args.validation_audit, "--updates", args.updates,
            "--experiment-scope", "bounded_refinement_experiment", "--paired-grounding-probability", args.grounding_probability, "--run"])
        status("completed_diagnostic_only", "done")
    except BaseException as exc:
        status("stopped_no_silent_retry", "failure", str(exc))
        raise


if __name__ == "__main__":
    main()
