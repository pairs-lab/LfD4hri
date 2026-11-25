"""ALL declared TRAIN text pairs: causal label/sensor-history audit, no model."""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.paired_decision_audit import paired_decision_frame
from VLA.smolvla.audit_target_grounding import paired_groups
from VLA.smolvla.audit_conditioning import verify_explicit_source_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_explicit_source_audit(args.source, args.source_audit)
    manifest = json.loads((args.source/"dataset_manifest.json").read_text())
    if manifest.get("text_intervention") != "target" or manifest.get("final_test_used") is not False:
        raise ValueError("Explicit ordinary TRAIN target-text intervention only")
    groups = paired_groups(json.loads((args.source/"index.json").read_text()))
    args.output.mkdir(parents=True)
    plan = {"scope": "ALL-source-pair causal decision/sensor audit_NOT_neural_grounding_or_success",
        "source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source/"index.json"),
        "source_audit_sha256": sha256_file(args.source_audit), "horizon": 5,
        "selected_pairs": [[r["file"] for r in pair] for _, pair in groups],
        "selection": "ALL pairs, first full-7D action divergence from identical complete sensor history",
        "selection_uses_neural_outcomes": False, "weights_loaded": False, "training_performed": False,
        "final_test_used": False, "source_already_in_new_fit_train_split": False,
        "helper_sha256": sha256_file(Path(__file__).with_name("paired_decision_audit.py"))}
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    rows = []
    for key, pair in groups:
        try:
            with np.load(args.source/pair[0]["file"], allow_pickle=False) as first, \
                    np.load(args.source/pair[1]["file"], allow_pickle=False) as second:
                decision = paired_decision_frame(first, second, horizon=5)
                reset_horizon = min(5, len(first["canonical_actions"]), len(second["canonical_actions"]))
                initial = first["canonical_actions"][:reset_horizon, :2]-second["canonical_actions"][:reset_horizon, :2]
                row = {"task": key[0], "files": [r["file"] for r in pair], "prompts": [r["task"]["prompt"] for r in pair],
                    "passed": True, **decision, "reset_teacher_xy_separation_rms_m": float(np.sqrt(np.mean(np.sum(initial**2, axis=1))))}
        except (ValueError, KeyError) as exc:
            row = {"task": key[0], "files": [r["file"] for r in pair], "passed": False,
                   "error": str(exc), "failure_preserved_not_filtered": True}
        rows.append(row)
        (args.output/"progress.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(len(rows), "/", len(groups), key[0], row["passed"], row.get("frame"), flush=True)
    verify_explicit_source_audit(args.source, args.source_audit)
    summaries = {task: {"pairs": sum(r["task"] == task for r in rows),
        "passed": sum(r["task"] == task and r["passed"] for r in rows),
        "informative_decisions": sum(r["task"] == task and r.get("informative_teacher_xy_separation", False) for r in rows),
        "uninformative_resets": sum(r["task"] == task and r.get("reset_teacher_xy_separation_rms_m", 1) < .002 for r in rows)}
        for task in ("touch", "pick", "move", "pick_place")}
    passed = all(r["passed"] for r in rows)
    (args.output/"results.json").write_text(json.dumps({**plan, "passed": passed,
        "summaries": summaries, "rows": rows}, indent=2)+"\n")
    if not passed:
        raise RuntimeError("Causal sensor/label failures preserved; do not shift frames or filter pairs")


if __name__ == "__main__":
    main()
