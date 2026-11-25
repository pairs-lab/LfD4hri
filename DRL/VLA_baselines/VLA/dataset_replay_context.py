"""SOURCE-AUDIT ONLY: exact reset context for native ground-truth replay.

Never imported by a learned policy. Paired target IDs/teacher poses are dataset
metadata and must not become runtime model inputs or controller fallbacks.
"""
import json
from pathlib import Path

from VLA.common.environment import ManipulationEnv
from VLA.common.observation_profile import ProfiledManipulationEnv
from VLA.common.tasks import TaskSpec
from VLA.common.environment_factory import environment_class_for_profile


def audit_environment(source, **kwargs):
    manifest = json.loads((Path(source)/"dataset_manifest.json").read_text())
    profile = manifest.get("observation_profile")
    if profile is None:
        return ManipulationEnv(**kwargs)
    return environment_class_for_profile(profile)(profile, **kwargs)


def reset_source_episode(env, source, item):
    """Replay historical labels with their recorded thresholds, not mutable XML.

    Only this OFFLINE audit helper pins distances. It deliberately retains the
    caller's requested streak length (e.g. first-hit replay). Learned policies
    and benchmarks do not import or call this module.
    """
    if item.get("paired_reset_strategy") != "canonical_spawn_order_then_teacher_retarget":
        observation = env.reset(TaskSpec.from_dict(item["task"]))
    else:
        from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment
        from VLA.audit_training_readiness import source_reset_provenance
        provenance = source_reset_provenance(source, item)
        canonical = next(row for row in json.loads((Path(provenance["canonical_source"])/"index.json").read_text())
                         if row["file"] == item["paired_scene_source"])
        wrapper = CanonicalSceneResetEnvironment(env, TaskSpec.from_dict(canonical["task"]),
                                                item.get("recovery_start_actions_not_training_labels", ()))
        observation = wrapper.reset(TaskSpec.from_dict(item["task"]))
    if item.get("success_contract"):
        from VLA.common.success_contract_snapshot import recorded_distance_thresholds
        env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(item)
    return observation
