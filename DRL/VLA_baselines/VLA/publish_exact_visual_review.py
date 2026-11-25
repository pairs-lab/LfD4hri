"""Carry an ACTUAL prior image review forward only for byte-identical outputs.

This does not judge images, qualify data, or bypass any workflow stage. Root
must inspect the previews and write the pre-review via apply_patch. A finite
watcher waits for the original workflow's review barrier. Different images,
reports, source hashes, or stopped workflows cause a hard failure, not retry.
"""
import argparse
import json
from pathlib import Path
import time

from VLA.common.provenance import sha256_file
from VLA.prepare_v9_candidate_fit import require_visual_receipt


def publish(workflow, pre_review):
    workflow, pre_review = Path(workflow).resolve(), Path(pre_review).resolve()
    review = json.loads(pre_review.read_text())
    status = json.loads((workflow/"workflow_status.json").read_text())
    if status.get("status") != "awaiting_actual_visual_review":
        raise ValueError("Workflow has not reached its audited visual-review barrier")
    if (review.get("reviewer") != "codex_actual_image_inspection"
            or not review.get("acceptable_for_bounded_fit")
            or set(review.get("tasks_reviewed", [])) != {"touch", "pick", "move", "pick_place"}
            or not review.get("byte_identical_republication_approved")):
        raise ValueError("No actual four-task inspection with explicit exact-byte approval")
    source = workflow/"combined_candidate"
    native = workflow/"native_smol512"
    audit = workflow/"combined_source_audit/audit.json"
    required = {"source_index_sha256": sha256_file(source/"index.json"),
        "native_report_sha256": sha256_file(native/"audit.json"),
        "source_audit_sha256": sha256_file(audit)}
    if any(review.get(key) != value for key, value in required.items()):
        raise ValueError("Actual preview review and final audit reports differ")
    images = []
    for item in review["images"]:
        inspected = Path(item["path"])
        final = workflow/item["final_relative_path"]
        if (final.resolve().is_relative_to(workflow) is False
                or sha256_file(inspected) != item["sha256"]
                or sha256_file(final) != item["sha256"]):
            raise ValueError("Final pixels are not byte-identical to the actually inspected image")
        images.append({"path": str(final.resolve()), "sha256": item["sha256"]})
    receipt = {**required, "reviewer": review["reviewer"], "tasks_reviewed": review["tasks_reviewed"],
        "acceptable_for_bounded_fit": True, "images": images,
        "actual_pre_review": str(pre_review), "actual_pre_review_sha256": sha256_file(pre_review),
        "review_transfer": "All final images and native/source reports match the inspected previews exactly; NOT automatic image approval",
        "known_limits": review.get("known_limits", [])}
    target, pending = workflow/"visual_review_receipt.json", workflow/".pending_exact_visual_review.json"
    if target.exists() or pending.exists():
        raise FileExistsError("Never overwrite a review receipt")
    pending.write_text(json.dumps(receipt, indent=2)+"\n")
    require_visual_receipt(pending, source, native, audit)
    # Atomic visibility: workflow cannot read a half-written receipt.
    pending.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--pre-review", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=10800)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 14400:
        raise ValueError("Use a finite review wait")
    digest = sha256_file(args.pre_review)
    deadline = time.monotonic()+args.wait_seconds
    while time.monotonic() < deadline:
        if sha256_file(args.pre_review) != digest:
            raise RuntimeError("Actual pre-review changed during wait")
        status = json.loads((args.workflow/"workflow_status.json").read_text())
        if status["status"] == "stopped_no_silent_retry":
            raise RuntimeError("Audit workflow stopped; no visual receipt or training authorization published")
        if status["status"] == "awaiting_actual_visual_review":
            print(str(publish(args.workflow, args.pre_review)), flush=True)
            return
        time.sleep(20)
    raise TimeoutError("No audited visual-review barrier within finite wait; no receipt published")


if __name__ == "__main__":
    main()
