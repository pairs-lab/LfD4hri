"""Immutable ALL-ROWS merge for new audits, never a training promotion.

Correction copies do not count as independent layouts. Canonical reset and
missed-close prefixes remain reset-only, never exposed as policy inputs.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from VLA.audit_training_readiness import physical_scene_key, source_reset_provenance
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint


def merge(sources, validation, output):
    sources = [Path(p).resolve() for p in sources]
    output, validation = Path(output).resolve(), Path(validation).resolve()
    if output.exists() or not sources or len(set(sources)) != len(sources) or validation in sources:
        raise ValueError("Use a new output and unique TRAIN components, never VALIDATION")
    val_index = validation/"index.json"
    val_hash = sha256_file(val_index)
    excluded = {physical_scene_key(row["task"]) for row in json.loads(val_index.read_text())}
    shared = ("robot", "robot_urdf", "gripper_asset", "image_size", "physics_hz", "control_hz",
              "required_success_steps", "observation_profile_sha256", "environment_runtime_fingerprint",
              "gripper_state_contract", "native_gripper_action_kind", "native_joint_action_kind")
    manifest, components, rows, links = None, [], [], []
    cache = {}
    for number, source in enumerate(sources):
        parent = json.loads((source/"dataset_manifest.json").read_text())
        if parent.get("split") in {"val", "validation", "test"} or parent.get("final_test_used"):
            raise ValueError("Validation/TEST component forbidden")
        state = source/"collection_state.json"
        if state.exists() and not json.loads(state.read_text())["status"].startswith("completed"):
            raise ValueError("An incomplete/stopped collector cannot become merged labels")
        verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
        if manifest is None:
            manifest = copy.deepcopy(parent)
        elif any(parent.get(key) != manifest.get(key) for key in shared):
            raise ValueError("Component changes the sensor/robot/action/feedback contract")
        index = json.loads((source/"index.json").read_text())
        if not index or len({row["file"] for row in index}) != len(index):
            raise ValueError("Empty or duplicate component index")
        component = {"source": str(source), "index_sha256": sha256_file(source/"index.json"),
                     "manifest_sha256": sha256_file(source/"dataset_manifest.json"), "episodes": len(index), "payloads": {}}
        for original in index:
            if not original["success"] or original.get("safety_violation"):
                raise ValueError("Failed/unsafe episode is preserved, never filtered into training")
            if physical_scene_key(original["task"]) in excluded:
                raise ValueError("A component overlaps frozen comparison VALIDATION")
            payload = source/original["file"]
            digest = sha256_file(payload)
            component["payloads"][original["file"]] = digest
            filename = f"component{number}_{original['file']}"
            reset = source_reset_provenance(source, original, cache)
            rows.append({**original, **reset, "file": filename, "candidate_only": True,
                         "merge_origin": {"source": str(source), "file": original["file"], "sha256": digest}})
            if parent.get("text_intervention") == "target":
                rows[-1]["grounding_text_intervention"] = "target"
            links.append((payload, filename))
        components.append(component)
    if sha256_file(val_index) != val_hash:
        raise ValueError("Comparison validation changed during declaration")
    output.mkdir(parents=True)
    for payload, filename in links:
        os.link(payload, output/filename)
    for component in components:
        parent_source = Path(component["source"])
        if (sha256_file(parent_source/"index.json") != component["index_sha256"]
                or sha256_file(parent_source/"dataset_manifest.json") != component["manifest_sha256"]
                or any(sha256_file(parent_source/file) != digest for file, digest in component["payloads"].items())):
            raise RuntimeError("Component changed during merge; output remains unqualified")
    (output/"index.json").write_text(json.dumps(rows, indent=2)+"\n")
    for key in ("split", "readiness_report", "readiness_report_sha256", "training_qualification",
                "qualification_parent", "qualification_parent_index_sha256", "qualification_parent_manifest_sha256",
                "collection_plan_sha256", "camera_report_sha256", "paired_audit_sha256"):
        manifest.pop(key, None)
    manifest.update(episodes=len(rows), frames=sum(row["steps"] for row in rows), episodes_per_task=None,
        declared_task_counts=dict(Counter(row["task"]["name"] for row in rows)), stratified_target_vocabulary=False,
        candidate_only=True, training_certified=False, final_test_used=False, immutable_hardlinked_payloads=True,
        purpose="All declared independent/corrective TRAIN components; NOT qualified or sufficient data",
        merge_components=components, index_sha256=sha256_file(output/"index.json"),
        comparison_validation={"source": str(validation), "index_sha256": val_hash},
        independent_layouts=len({physical_scene_key(row["task"]) for row in rows}),
        correction_copies_count_as_independent_layouts=False)
    (output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return {"source": str(output), "episodes": len(rows), "task_counts": manifest["declared_task_counts"],
            "independent_layouts": manifest["independent_layouts"], "training_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.validation, args.output), indent=2))


if __name__ == "__main__":
    main()
