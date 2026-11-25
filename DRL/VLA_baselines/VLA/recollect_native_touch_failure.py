"""Recollect an exact pre-fit touch scene with a force-bearing contact margin.

Candidate/engineering evidence only. No runtime, success, camera or original
payload changes. Preserve the failed original, and independently replay the
new labels before any replacement/promotion can be considered.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchCollection


def require_same_reset(arrays, reference):
    keys = ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure")
    errors = {key: float(np.max(np.abs(np.asarray(arrays[key][0], float) -
                                    np.asarray(reference[key], float)))) for key in keys}
    if any(errors.values()):
        raise ValueError("Recollection changed original sensor reset: " + str(errors))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Never overwrite failed evidence or prior repair")
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    row = next(r for r in json.loads((source / "index.json").read_text()) if r["file"] == args.episode)
    if (row["task"]["name"] != "touch" or manifest.get("candidate_only") is not True
            or manifest.get("final_test_used") or row.get("final_test_used")
            or row.get("paired_scene_source")):
        raise ValueError("Only an ordinary pre-fit touch candidate can be recollected here")
    protected = {str(path): sha256_file(path) for path in
                 (source / "index.json", source / "dataset_manifest.json", source / args.episode,
                  Path(__file__).resolve(), Path(__file__).with_name("staged_acquisition_teacher.py"),
                  Path(__file__).with_name("surface_limited_touch_teacher.py"))}
    with np.load(source / args.episode, allow_pickle=False) as old:
        reference = {key: old[key][0].copy() for key in
                     ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure")}
    output.mkdir(parents=True)
    with audit_environment(source, image_size=384, required_success_steps=1,
                           use_semantic_success=False) as env:
        proxy = SurfaceLimitedTouchCollection(env, source, row)
        arrays, metadata = collect_episode(proxy, TaskSpec.from_dict(row["task"]),
                                          policy_factory=StagedAcquisitionTeacher)
        force, quality = proxy.last_contact_force, proxy.last_quality_failure
    reset_errors = require_same_reset(arrays, reference)
    repaired = copy.deepcopy(row)
    repaired.update(metadata, candidate_only=True,
                    expert_terminal_requires_positive_contact=True,
                    expert_required_positive_contact_ticks=2,
                    expert_terminal_force_quality_cap_n=5.,
                    expert_terminal_max_normal_force_n=force,
                    expert_quality_failure=quality,
                    native_touch_recollection_origin={"source": str(source), "file": args.episode,
                                                      "sha256": protected[str(source / args.episode)]})
    repaired.pop("merge_origin", None)
    np.savez_compressed(output / args.episode, **arrays)
    (output / "index.json").write_text(json.dumps([repaired], indent=2) + "\n")
    candidate = copy.deepcopy(manifest)
    for key in ("merge_components", "comparison_validation", "collection_plan_sha256",
                "independent_layouts", "split", "training_qualification", "readiness_report",
                "readiness_report_sha256"):
        candidate.pop(key, None)
    candidate.update(candidate_only=True, training_certified=False,
                     stratified_target_vocabulary=False, episodes=1, frames=metadata["steps"],
                     declared_task_counts={"touch": 1}, episodes_per_task=None,
                     purpose="Exact same-scene force-bearing touch candidate; not qualified training",
                     index_sha256=sha256_file(output / "index.json"))
    (output / "dataset_manifest.json").write_text(json.dumps(candidate, indent=2) + "\n")
    passed = metadata["success"] and not metadata.get("safety_violation") and quality is None
    report = {"passed": bool(passed), "original_episode": args.episode, "source": str(source),
              "reset_errors": reset_errors, "metadata": repaired, "protected_files": protected,
              "source_payload_fingerprint": source_payload_fingerprint(output),
              "runtime_changed": False, "success_changed": False,
              "original_failure_preserved": True, "training_performed": False,
              "physics_replay_still_required": True, "final_test_used": False}
    if any(sha256_file(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError("Original data or collector changed during recollection")
    (output / "recollection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": bool(passed), "steps": metadata["steps"],
                      "force_n": force, "quality_failure": quality,
                      "reset_errors": reset_errors}), flush=True)
    if not passed:
        raise RuntimeError("Preserved unsuccessful recollection; do not train")


if __name__ == "__main__":
    main()
