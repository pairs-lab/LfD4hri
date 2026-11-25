"""Keep ALL completed scenes; replace precisely the audited bad-observation paths.

Never removes episodes, edits originals, changes cameras/runtime/scoring, or
qualifies training. Requires complete-source visibility and actual BOTH strict
physics proofs for every replacement. Safe unchanged payloads are hardlinked.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import time

from VLA.common.provenance import sha256_file


def validated_replacements(visibility, repair, replay):
    expected = {row["file"] for row in visibility["rows"] if row["failures"]}
    supplied = {row["file"] for row in repair["rows"]}
    if (not visibility["all_episodes"] or not expected or supplied != expected
            or not repair.get("passed") or repair.get("runtime_changed") is not False
            or repair.get("cameras_changed") is not False or repair.get("training_performed") is not False
            or not all(row["passed"] and not any(row["reset_errors"].values()) for row in repair["rows"])):
        raise ValueError("Needs exact ALL full-source failures repaired on identical resets")
    if (replay.get("passed") is not True or replay.get("mode") != "both"
            or replay.get("native_success_mode") != "strict" or replay.get("failures")
            or any(replay["summaries"][mode]["episodes"] != len(expected)
                   or replay["summaries"][mode]["successes"] != len(expected)
                   or replay["summaries"][mode]["initial_false_positives"] != 0
                   for mode in ("canonical", "native"))):
        raise ValueError("ALL strict canonical/native physics must pass")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "visibility-report", "repair", "repair-replay", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.wait_seconds <= 3600:
        raise ValueError("Fresh output and finite report wait required")
    deadline = time.monotonic() + args.wait_seconds
    required_reports = [args.visibility_report, args.repair/"results.json", args.repair_replay]
    while not all(path.is_file() for path in required_reports):
        if time.monotonic() >= deadline:
            raise TimeoutError("Complete visibility proof missing; no dataset replacement")
        time.sleep(10)
    source, repair = args.source.resolve(), args.repair.resolve()
    visibility = json.loads(args.visibility_report.read_text())
    evidence = json.loads((repair/"results.json").read_text())
    replay = json.loads(args.repair_replay.read_text())
    selected = validated_replacements(visibility, evidence, replay)
    rows = json.loads((source/"index.json").read_text())
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    counts = Counter(row["task"]["name"] for row in rows)
    if (state["status"] != "completed_candidate_not_qualified" or state["completed_episodes"] != 2280
            or counts != {name: 570 for name in ("touch", "pick", "move", "pick_place")}
            or Path(visibility["source"]).resolve() != source
            or visibility["source_index_sha256"] != sha256_file(source/"index.json")
            or visibility["source_manifest_sha256"] != sha256_file(source/"dataset_manifest.json")
            or Path(evidence["source"]).resolve() != source
            or Path(replay["source"]).resolve() != repair
            or replay["source_index_sha256"] != sha256_file(repair/"index.json")
            or replay["source_manifest_sha256"] != sha256_file(repair/"dataset_manifest.json")):
        raise ValueError("Complete parent or strict replacement proof changed")
    for path, digest in evidence["frozen_files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Pinned repair evidence/implementation changed")
    replacements = {row["file"]: row for row in json.loads((repair/"index.json").read_text())}
    if set(replacements) != selected or any(not row["success"] or row.get("safety_violation") for row in rows):
        raise ValueError("Missing replacement or unsafe original physical scene")
    preserved, new_rows = {}, []
    for old in rows:
        filename = old["file"]
        digest = sha256_file(source/filename)
        if filename in selected:
            fresh = replacements[filename]
            origin = fresh["transport_repair_origin"]
            if (fresh["task"] != old["task"] or fresh["scene_group"] != old["scene_group"]
                    or Path(origin["source"]).resolve() != source or origin["sha256"] != digest
                    or not fresh["success"] or fresh.get("safety_violation")):
                raise ValueError("Replacement altered an exact task/scene or lacks safe execution")
            new_rows.append({**old, **fresh})
        else:
            new_rows.append(old)
        preserved[filename] = digest
    args.output.mkdir(parents=True)
    frozen = {str(path.resolve()): sha256_file(path) for path in
        (source/"index.json", source/"dataset_manifest.json", source/"collection_state.json",
         args.visibility_report, repair/"results.json", repair/"index.json", args.repair_replay, Path(__file__))}
    for row in rows:
        filename = row["file"]
        payload = repair/filename if filename in selected else source/filename
        os.link(payload, args.output/filename)
    (args.output/"index.json").write_text(json.dumps(new_rows, indent=2)+"\n")
    plan = {"scope": "ALL2280 original scenes, precisely audited visibility paths recollected; NOT qualified TRAIN",
        "source": str(source), "repair": str(repair), "frozen_files": frozen,
        "original_payload_sha256": preserved, "replaced_files": sorted(selected),
        "unchanged_payloads": len(rows)-len(selected), "episodes_removed": 0,
        "runtime_changed": False, "cameras_changed": False, "scoring_changed": False,
        "training_authorized": False, "final_test_used": False}
    (args.output/"repair_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    result_manifest = copy.deepcopy(manifest)
    result_manifest.pop("collection_plan_sha256", None)
    result_manifest.update(candidate_only=True, training_certified=False, episodes=len(new_rows),
        frames=sum(row["steps"] for row in new_rows), index_sha256=sha256_file(args.output/"index.json"),
        repair_plan_sha256=sha256_file(args.output/"repair_plan.json"))
    (args.output/"dataset_manifest.json").write_text(json.dumps(result_manifest, indent=2)+"\n")
    if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
        raise RuntimeError("Parent/proof changed while materializing; do not use this output")
    (args.output/"collection_state.json").write_text(json.dumps({"status": "completed_candidate_not_qualified",
        "completed_episodes": len(new_rows), "planned_episodes": len(rows),
        "index_sha256": result_manifest["index_sha256"], "training_authorized": False,
        "replaced_files": sorted(selected)}, indent=2)+"\n")
    print(json.dumps({"episodes": len(new_rows), "replacement_count": len(selected),
        "unchanged_payloads": len(rows)-len(selected), "training_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
