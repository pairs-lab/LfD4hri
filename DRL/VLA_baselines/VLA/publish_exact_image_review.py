"""Publish a visual receipt only for bytes previously inspected by root.

This never inspects/approves unseen images, starts training, changes a recipe,
or skips workflow gates. A separately exported native preview can be reviewed
while full physics replay runs. Publication requires the workflow's eventual
native report and mandatory PNGs to be BYTE-IDENTICAL to that reviewed export.
Changed bytes, source, failed gates or missing approval stop publication.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import verify_source_payload_fingerprint
from VLA.prepare_v9_candidate_fit import require_visual_receipt

TASKS = {"touch", "pick", "move", "pick_place"}
REQUIRED_STAGES = {"merge", "combined_source_audit", "transport_visibility",
    "full_physics_replay", "native_camera", "readiness", "materialize_train",
    "materialize_validation", "train_candidate_audit", "validation_audit"}


def verified_receipt(approval, workflow, requirements, state):
    workflow = Path(workflow).resolve()
    if (approval.get("reviewer") != "codex_actual_image_inspection"
            or approval.get("scope") != "root_review_of_existing_exact_bytes"
            or approval.get("acceptable_for_bounded_fit") is not True
            or set(approval.get("tasks_reviewed", [])) != TASKS):
        raise ValueError("Missing actual root four-task approval")
    if (state.get("status") != "awaiting_actual_visual_review" or state.get("stage") != "review"
            or state.get("final_test_used") or not state.get("run_training_requested")):
        raise ValueError("Workflow is not at the successful pre-fit review barrier")
    completed = {r["stage"] for r in state["history"] if r.get("exit_code") == 0}
    if not REQUIRED_STAGES <= completed or any(r.get("exit_code") != 0 for r in state["history"]):
        raise ValueError("An upstream qualification stage is missing or failed")
    source = workflow/"combined_candidate"
    native = workflow/"native_smol512"
    source_audit = workflow/"combined_source_audit/audit.json"
    mandatory = {str(native/"native_inputs_contact_sheet.png"),
                 str(source_audit.parent/"camera_contact_sheet.png")}
    if (Path(approval["workflow"]).resolve() != workflow
            or requirements.get("source") != str(source)
            or set(requirements.get("required_images", [])) != mandatory
            or requirements.get("receipt") != str(workflow/"visual_review_receipt.json")):
        raise ValueError("Approval or review requirements refer to another workflow")
    bound = {
        "source_index_sha256": source/"index.json",
        "source_manifest_sha256": source/"dataset_manifest.json",
        "source_audit_sha256": source_audit,
        "native_report_sha256": native/"audit.json",
        "workflow_plan_sha256": workflow/"workflow_plan.json",
    }
    for key, path in bound.items():
        if sha256_file(path) != approval.get(key):
            raise ValueError("Approved bytes changed: "+key)
    for path, digest in approval.get("bound_files", {}).items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Inspected preprocessing/configuration changed: "+path)
    for key in ("source_index_sha256", "source_audit_sha256", "native_report_sha256"):
        if requirements.get(key) != approval[key]:
            raise ValueError("Workflow requirements disagree with root approval: "+key)
    report = json.loads(source_audit.read_text())
    replay = json.loads((workflow/"combined_full_physics_replay.json").read_text())
    if not report.get("passed") or report.get("failures"):
        raise ValueError("Actual complete source audit did not pass")
    if (not replay.get("passed") or replay.get("failures")
            or replay.get("all_episodes") is not True or replay.get("native_success_mode") != "strict"
            or any(replay.get("summaries", {}).get(mode, {}).get("episodes") != report["episodes"]
                   or replay["summaries"][mode].get("successes") != report["episodes"]
                   for mode in ("canonical", "native"))):
        raise ValueError("Complete canonical/native strict replay is not successful")
    image_rows = approval.get("images", [])
    if not mandatory <= {r.get("workflow_path") for r in image_rows}:
        raise ValueError("Root did not review both mandatory image contents")
    images = []
    for row in image_rows:
        original, destination = Path(row["inspected_path"]), Path(row["workflow_path"])
        if sha256_file(original) != row["sha256"] or sha256_file(destination) != row["sha256"]:
            raise ValueError("Workflow image is not byte-identical to root-inspected image")
        images.append({"path": str(destination), "sha256": row["sha256"],
                       "original_inspected_path": str(original)})
    verify_source_payload_fingerprint(report["source_payload_fingerprint"], source)
    return {"reviewer": approval["reviewer"], "tasks_reviewed": sorted(TASKS),
        "acceptable_for_bounded_fit": True,
        **{key: approval[key] for key in ("source_index_sha256", "native_report_sha256", "source_audit_sha256")},
        "images": images, "review_method": "root actual inspection plus subsequent exact-byte equivalence, NOT unseen image approval",
        "source_manifest_sha256": approval["source_manifest_sha256"],
        "known_limits": approval.get("known_limits", []),
        "data_sufficiency_established": False, "learned_success_claimed": False,
        "published_at_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.wait_seconds <= 86400:
        raise ValueError("Use a fresh monitor output and a finite wait")
    workflow = args.workflow.resolve()
    args.output.mkdir(parents=True)
    approval_hash = sha256_file(args.approval)
    implementation = sha256_file(Path(__file__))
    approval = json.loads(args.approval.read_text())
    status_path = args.output/"status.json"
    def status(value, error=None):
        status_path.write_text(json.dumps({"status": value, "error": error, "pid": os.getpid(),
            "workflow": str(workflow), "approval": str(args.approval.resolve()),
            "approval_sha256": approval_hash, "implementation_sha256": implementation,
            "receipt_published": value == "published_exact_byte_review_receipt",
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
    receipt = workflow/"visual_review_receipt.json"
    pending = workflow/".exact_byte_review_receipt.pending.json"
    status("waiting_for_all_gates_and_exact_images")
    try:
        deadline = time.monotonic()+args.wait_seconds
        while True:
            if sha256_file(args.approval) != approval_hash or sha256_file(Path(__file__)) != implementation:
                raise RuntimeError("Approval or publisher changed during wait")
            if receipt.exists():
                raise FileExistsError("Do not overwrite another visual review")
            try:
                state = json.loads((workflow/"workflow_status.json").read_text())
            except json.JSONDecodeError:
                state = None
            if state and state.get("status") == "stopped_no_silent_retry":
                raise RuntimeError("Upstream stopped; no approval publication: "+str(state.get("error")))
            if state and state.get("status") == "awaiting_actual_visual_review":
                requirements = json.loads((workflow/"visual_review_requirements.json").read_text())
                payload = verified_receipt(approval, workflow, requirements, state)
                payload["approval_sha256"] = approval_hash
                with pending.open("x") as stream:
                    stream.write(json.dumps(payload, indent=2)+"\n")
                require_visual_receipt(pending, workflow/"combined_candidate", workflow/"native_smol512",
                    workflow/"combined_source_audit/audit.json")
                os.link(pending, receipt)  # atomic completed file; never overwrite
                pending.unlink()  # only this publisher's generated temporary artifact
                status("published_exact_byte_review_receipt")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("No completed qualification/review barrier within bounded wait")
            time.sleep(20)
    except Exception as exc:
        status("stopped_without_training_authorization", str(exc))
        raise


if __name__ == "__main__":
    main()
