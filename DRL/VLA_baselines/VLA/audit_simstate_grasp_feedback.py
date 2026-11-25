"""Quantify current bilateral feedback versus current target attachment.

Before-state references use t-1 recorded outcomes, never t future labels as
policy input. No weight/controller/data changes or neural success estimate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.smolvla.simstate_dataset import load_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep every feedback gap audit")
    states, proof, _ = load_projection(args.projection)
    source = Path(proof["source"]); offset = 0; counts = Counter(); rows = []
    for record in proof["rows"]:
        length = record["frames"]; current = states[offset:offset+length]; offset += length
        with np.load(source/record["file"], allow_pickle=False) as arrays:
            # All qualified source starts are independently NOT-held. The
            # prior AFTER outcome is current BEFORE attachment, not future.
            attached = np.r_[False,arrays["next_grasped_object"][:-1]].astype(bool)
            bilateral = current[:, 17].astype(bool)
            reference = np.r_[False,arrays["next_verified_grasp"][:-1]].astype(bool)
            if not np.array_equal(bilateral,reference):
                raise ValueError("Before-state contact feedback disagrees with the prior recorded physical outcome")
            if np.any(bilateral & ~attached):
                raise ValueError("Advertised verified target contact without actual target attachment")
            gap = attached & ~bilateral
            labels = arrays["canonical_actions"]
            closing_lift = (labels[:,6] > .2) & (labels[:,2] > .005)
            summary = {"frames":length,"correct_target_attached_before":int(attached.sum()),
                "verified_bilateral_before":int(bilateral.sum()),"attached_with_bilateral_feedback_zero":int(gap.sum()),
                "teacher_close_lift_with_bilateral_feedback_zero":int((closing_lift & ~bilateral).sum()),
                "teacher_close_lift_with_NO_target_attachment":int((closing_lift & ~attached).sum())}
        counts.update(summary); rows.append({"file":record["file"],"task":record["task"],**summary})
    atomic_json(args.output,{"passed":True,"scope":"full_TRAIN_causal_feedback_semantic_gap_NOT_model_SR",
        "projection_audit_sha256":sha256_file(args.projection/"audit.json"),"counts":dict(counts),"rows":rows,
        "bilateral_zero_is_NOT_equivalent_to_not_attached":True,"future_state_in_policy_input":False,
        "data_or_weights_or_controller_modified":False})
    print(dict(counts),flush=True)


if __name__ == "__main__":
    main()
