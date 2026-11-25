"""Recollect ALL existing near-miss states with a predeclared source contract.

Original scenes/startup controls/observations stay intact. Old arrays and
success labels are preserved; this creates genuinely new expert transitions,
not relabelled passes or a runtime teacher fallback.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.collect_demos import collect_episode
from VLA.collect_expanded_v9_candidate import frozen_thresholds
from VLA.collect_v9_recovery_candidate import GentleRecoveryCollection, SafeRecoveryOpeningTeacher, recovery_checks
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher


class FrozenRecoveryReset:
    """Reset pinning AFTER actual canonical startup, never a model observation.

    The retarget/reset helper acts on the real environment. Returning an
    already-captured reset avoids a forwarding proxy whose attribute writes
    might shadow actual task IDs or physical success state.
    """
    def __init__(self, env, source, row, thresholds, reference):
        self.env, self.source, self.row = env, source, row
        self.thresholds, self.reference = thresholds, reference
        self.reset_errors = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        # Dataclass tuples become lists in saved JSON. Normalize containers
        # through Python JSON only, preserving numeric 0.0 representations.
        if json.loads(json.dumps(task.to_dict())) != self.row["task"]:
            raise ValueError("Recollection changed original instruction/physical task")
        observation = reset_source_episode(self.env, self.source, self.row)
        self.env.dist_threshold, self.env.obj_threshold = self.thresholds[task.name]
        measured = {"front_rgb": observation.front_rgb, "wrist_rgb": observation.wrist_rgb,
                    "joint_positions": observation.joint_positions,
                    "gripper_closure": self.env.normalized_gripper_closure()}
        self.reset_errors = {key: float(np.max(np.abs(np.asarray(value, float) - np.asarray(self.reference[key], float))))
                             for key, value in measured.items()}
        if any(self.reset_errors.values()):
            raise ValueError("Exact pinned near-miss reset differs: " + str(self.reset_errors))
        return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "contract-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source, contract, output = (path.resolve() for path in (args.source, args.contract_source, args.output))
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads((source / "index.json").read_text())
    thresholds = frozen_thresholds(json.loads((contract / "index.json").read_text()))
    if (len(rows) != 152 or Counter(row["task"]["name"] for row in rows) !=
            {name: 38 for name in ("touch", "pick", "move", "pick_place")} or
            any(not row.get("near_object_recovery_start_evidence") or
                not row.get("recovery_start_actions_not_training_labels") for row in rows)):
        raise ValueError("Requires ALL exact152 real near-body recovery states")
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    profile = manifest["observation_profile"]
    verify_environment_runtime_fingerprint(manifest["environment_runtime_fingerprint"], profile)
    plan_parent = json.loads((source / "collection_plan.json").read_text())
    frozen = dict(plan_parent["offline_label_sources_sha256"])
    frozen.update({str(path): sha256_file(path) for path in
                   (Path(__file__).resolve(), source / "index.json", source / "dataset_manifest.json",
                    source / "collection_plan.json", contract / "index.json", contract / "dataset_manifest.json")})
    payload = source_payload_fingerprint(source, rows)
    def verify():
        if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
            raise RuntimeError("Frozen recollection source/teacher/contract changed")
        verify_environment_runtime_fingerprint(manifest["environment_runtime_fingerprint"], profile)
    verify()
    output.mkdir(parents=True)
    plan = {"source": str(source), "contract_source": str(contract), "frozen_files": frozen,
            "source_payload_fingerprint": payload, "frozen_reward_distances": thresholds,
            "planned_episodes": 152, "selected_files": [row["file"] for row in rows],
            "scope": "ALL original near-body states; new expert labels under uniform frozen distances",
            "original_arrays_relabelled": False, "scenes_removed": 0,
            "runtime_or_cameras_changed": False, "training_performed": False, "final_test_used": False,
            "pybullet_binary_sha256": sha256_file(Path(p.__file__))}
    (output / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    state = {"status": "in_progress", "planned_episodes": 152, "completed_episodes": 0,
             "training_performed": False, "error": None}
    def save_state():
        (output / "collection_state.json").write_text(json.dumps(state, indent=2) + "\n")
    save_state()
    collected, proofs = [], []
    try:
        with audit_environment(source, image_size=384, required_success_steps=1,
                               success_threshold_scale=1., use_semantic_success=False) as env:
            for row in rows:
                verify()
                with np.load(source / row["file"], allow_pickle=False) as old:
                    reference = {key: old[key][0].copy() for key in
                                 ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure")}
                proxy = FrozenRecoveryReset(env, source, row, thresholds, reference)
                task = TaskSpec.from_dict(row["task"])
                collection = GentleRecoveryCollection(proxy) if task.name == "touch" else proxy
                arrays, metadata = collect_episode(collection, task,
                    policy_factory=lambda wrapped: SafeRecoveryOpeningTeacher(wrapped, YawClearTouchTeacher))
                checks = recovery_checks(arrays, task.name, metadata)
                preserved = copy.deepcopy(row)
                preserved.update(metadata, file=row["file"], candidate_only=True,
                    near_object_recovery_checks=checks,
                    frozen_recollection_origin={"source": str(source), "file": row["file"],
                                                "sha256": sha256_file(source / row["file"])})
                if task.name == "touch":
                    gap, force = collection.terminal_geometry()
                    preserved.update(expert_terminal_surface_gap_m=gap,
                                     expert_terminal_max_normal_force_n=force,
                                     force_verified_physical_contact_claimed=False)
                np.savez_compressed(output / row["file"], **arrays)
                collected.append(preserved)
                proofs.append({"file": row["file"], "reset_errors": proxy.reset_errors,
                               "checks": checks, "success": metadata["success"]})
                (output / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
                (output / "recollection_proofs.json").write_text(json.dumps(proofs, indent=2) + "\n")
                state["completed_episodes"] = len(collected)
                save_state()
                print(len(collected), "/152", row["file"], metadata["success"], metadata["steps"], flush=True)
                if not metadata["success"] or not all(checks.values()):
                    raise RuntimeError("Failed recollection retained; no scene skipping or training")
        verify()
        verify_source_payload_fingerprint(payload, source, rows)
        candidate = copy.deepcopy(manifest)
        for key in ("split", "training_qualification", "readiness_report", "readiness_report_sha256",
                    "collection_plan_sha256", "qualification_parent"):
            candidate.pop(key, None)
        candidate.update(candidate_only=True, training_certified=False, episodes=len(collected),
                         frames=sum(row["steps"] for row in collected),
                         declared_task_counts=dict(Counter(row["task"]["name"] for row in collected)),
                         index_sha256=sha256_file(output / "index.json"),
                         collection_plan_sha256=sha256_file(output / "collection_plan.json"),
                         purpose="Uniform-frozen-contract near-miss recollection; NOT qualified TRAIN or learned SR")
        (output / "dataset_manifest.json").write_text(json.dumps(candidate, indent=2) + "\n")
        state.update(status="completed_candidate_not_qualified", index_sha256=candidate["index_sha256"])
        save_state()
    except Exception as exc:
        state.update(status="stopped_error_NOT_training", error=str(exc))
        save_state()
        raise


if __name__ == "__main__":
    main()
