"""Same preserved excluded TRAIN layout, baseline-aware start and two text corrections."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.audit_counterfactual_collection import alternate_target
from VLA.collect_demos import collect_episode
from VLA.collect_paired_grounding_candidate import pair_errors
from VLA.collect_v9_recovery_candidate import SafeRecoveryOpeningTeacher, recovery_checks
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.near_object_recovery_start import closed_empty_near_object_start, validated_near_object_reset_environment
from VLA.settling_baseline import baseline_object_poses
from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-surface-terminal-quality", action="store_true",
                        help="Offline strict success AND existing <=0.2mm/<=5N touch-label quality; no evaluation relaxation")
    parser.add_argument("--bounded-transit", action="store_true")
    parser.add_argument("--expanded-candidates", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    diagnosis = json.loads(args.diagnosis.read_text())
    if (not diagnosis["original_rejection_message_identical"] or diagnosis["training_performed"]
            or diagnosis["final_test_used"]):
        raise ValueError("Exact reproduced excluded TRAIN layout required")
    source = Path(diagnosis["source"])
    old_plan = json.loads((Path(diagnosis["failed_candidate"])/"collection_plan.json").read_text())
    row = next(r for r in json.loads((source/"index.json").read_text()) if r["file"] == diagnosis["source_episode"])
    task = TaskSpec.from_dict(row["task"])
    number = old_plan["selected_source_files"].index(row["file"])
    alternate = alternate_target(task, choice=number)
    args.output.mkdir(parents=True)
    protected = [source/"index.json", source/"dataset_manifest.json", source/row["file"], args.diagnosis,
        Path(__file__), Path(__file__).with_name("near_object_recovery_start.py"),
        Path(__file__).with_name("settling_baseline.py"), Path(__file__).with_name("yaw_clear_touch_teacher.py"),
        Path(__file__).with_name("near_object_candidate_expansion.py"),
        Path(__file__).with_name("collect_v9_recovery_candidate.py"),
        Path(__file__).with_name("semantic_touch_teacher.py")]
    hashes = {str(path.resolve()): sha256_file(path) for path in protected}
    rows, arrays_pair = [], []
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        def baseline(task, actions):
            return pool.submit(baseline_object_poses, (str(source), task.to_dict(), [float(a[6]) for a in actions])).result(timeout=120)
        with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
            reset_source_episode(env, source, row)
            startup, evidence = closed_empty_near_object_start(env, task, settling_baseline=baseline,
                bounded_transit=args.bounded_transit, expanded_candidates=args.expanded_candidates)
            proxy = validated_near_object_reset_environment(env, task, startup, evidence)
            collection = proxy
            if task.name == "touch" and args.near_surface_terminal_quality:
                from VLA.collect_v9_recovery_candidate import GentleRecoveryCollection
                collection = GentleRecoveryCollection(proxy)
            for variant, instruction in enumerate((task, alternate)):
                arrays, meta = collect_episode(collection, instruction,
                    policy_factory=lambda wrapper: SafeRecoveryOpeningTeacher(wrapper, YawClearTouchTeacher))
                filename = f"{task.name}_layout{number:03d}_variant{variant}.npz"
                checks = recovery_checks(arrays, task.name, meta)
                meta.update(file=filename, variant=variant, candidate_only=True, canonical_source=str(source),
                    canonical_source_index_sha256=sha256_file(source/"index.json"), paired_scene_source=row["file"],
                    paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                    recovery_start_actions_not_training_labels=[a.tolist() for a in startup],
                    recovery_start_kind="closed_empty_near_object_missed_close", recovery_checks=checks,
                    near_object_recovery_start_evidence=evidence)
                if task.name == "touch" and args.near_surface_terminal_quality:
                    gap, force = collection.terminal_geometry()
                    meta.update(gentle_near_touch=True, expert_terminal_surface_gap_m=gap,
                                expert_terminal_max_normal_force_n=force)
                np.savez_compressed(args.output/filename, **arrays)
                rows.append(meta); arrays_pair.append(arrays)
                (args.output/"index.json").write_text(json.dumps(rows, indent=2)+"\n")
                for frame in sorted(set([0, min(15, meta["steps"]-1), meta["steps"]-1])):
                    for camera in ("front", "wrist"):
                        Image.fromarray(arrays[camera+"_rgb"][frame]).save(args.output/f"variant{variant}_frame{frame:03d}_{camera}.png")
                print(variant, instruction.prompt, meta["success"], meta["steps"], checks, flush=True)
    manifest = copy.deepcopy(json.loads((source/"dataset_manifest.json").read_text()))
    for key in ("split", "training_qualification", "qualification_parent", "qualification_parent_index_sha256",
                "qualification_parent_manifest_sha256", "collection_plan_sha256"):
        manifest.pop(key, None)
    manifest.update(candidate_only=True, training_certified=False, episodes=len(rows),
        frames=sum(r["steps"] for r in rows), declared_task_counts={task.name: len(rows)},
        index_sha256=sha256_file(args.output/"index.json"), final_test_used=False)
    (args.output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    passed = all(r["success"] and not r.get("safety_violation") and all(r["recovery_checks"].values()) for r in rows)
    report = {"scope": "Same excluded TRAIN layout two-text near-body recovery teacher pilot_NOT_neural_success",
        "passed": passed, "source": str(source), "source_episode": row["file"], "rows": rows,
        "start_evidence": evidence, "same_sensor_text_pair": pair_errors(*arrays_pair),
        "protected_hashes": hashes, "training_performed": False, "final_test_used": False,
        "runtime_camera_robot_objects_success_safety_changed": False}
    report["offline_near_surface_touch_label_quality"] = args.near_surface_terminal_quality
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Preserved TRAIN or repair code changed during pilot")
    if not passed:
        raise RuntimeError("Repair pilot failed; retain, do not filter")


if __name__ == "__main__":
    main()
