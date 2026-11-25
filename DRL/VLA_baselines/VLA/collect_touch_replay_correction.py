"""Recollect one EXACT preserved TRAIN touch case; candidate only, no filtering."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.collect_demos import collect_episode
from VLA.collect_paired_grounding_candidate import FrozenRetargetReset
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.robust_touch_teacher import RobustTouchTeacher, GentleForceTouchTeacher


class ForceVerifiedTouchCollection(FrozenRetargetReset):
    """TEACHER ONLY: collect past proximity first-hit until light contact.

    Original reward/raw-success fields are preserved, not redefined. Training
    and learned-policy evaluation NEVER instantiate this wrapper. Timeouts and
    safety termination cannot be overridden. No live simulator flag is edited.
    """
    def __init__(self, env, source, row):
        super().__init__(env, source, row)
        self.verified_ticks = 0

    def reset(self, task):
        self.verified_ticks = 0
        return super().reset(task)

    def positive_contact(self):
        return any(c[9] > 1e-6 for c in p.getContactPoints(self.env.gripper.body, self.env.pick_id))

    @property
    def _last_success(self):
        # Avoid the teacher's HOLD on a zero-force cached contact record.
        return self.env._last_success and self.verified_ticks >= 2 and self.positive_contact()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.verified_ticks = self.verified_ticks + 1 if self.positive_contact() else 0
        verified = self.verified_ticks >= 2
        info = dict(info, expert_positive_contact_verified=verified)
        force = max((c[9] for c in p.getContactPoints(self.env.gripper.body, self.env.pick_id)), default=0.)
        self.last_contact_force = force
        self.last_quality_failure = "contact_force_above_predeclared_5N" if force > 5. else None
        info["expert_terminal_max_normal_force_n"] = force
        # Predeclared DATA quality cap; this does not modify runtime safety or
        # benchmark success. A stronger-force candidate is retained as failed.
        if force > 5.:
            return observation, reward, True, dict(info, success=False,
                expert_quality_failure="contact_force_above_predeclared_5N")
        if not verified:
            info["success"] = False
            if not info.get("safety_violation") and self.env.step_count < self.env.task.max_steps:
                done = False
        return observation, reward, done, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-positive-contact", action="store_true",
                        help="Expert-only stricter terminal quality check; no benchmark success changes")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    allowed = {row["file"]: row for row in json.loads(args.train_episodes.read_text())}
    if args.episode not in allowed or Path(allowed[args.episode]["source"]).resolve() != source:
        raise ValueError("No corrective labels from validation or final test")
    row = next(r for r in json.loads((source/"index.json").read_text()) if r["file"] == args.episode)
    if row["task"]["name"] != "touch" or sha256_file(source/args.episode) != allowed[args.episode]["sha256"]:
        raise ValueError("Changed or non-touch TRAIN episode")
    args.output.mkdir(parents=True)
    with audit_environment(source, image_size=384, required_success_steps=1,
                           use_semantic_success=False) as env:
        reset_class = ForceVerifiedTouchCollection if args.require_positive_contact else FrozenRetargetReset
        proxy = reset_class(env, source, row)
        arrays, meta = collect_episode(proxy, TaskSpec.from_dict(row["task"]),
                                       policy_factory=GentleForceTouchTeacher if args.require_positive_contact else RobustTouchTeacher)
        if args.require_positive_contact:
            meta.update(expert_terminal_max_normal_force_n=proxy.last_contact_force,
                        expert_quality_failure=proxy.last_quality_failure)
    with np.load(source/args.episode, allow_pickle=False) as original:
        if any(not np.array_equal(arrays[key][0], original[key][0]) for key in
               ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure")):
            raise RuntimeError("Correction changed original reset observations")
    filename = "touch_correction.npz"
    np.savez_compressed(args.output/filename, **arrays)
    # variant0 binds the existing canonical-reset protocol, not a fabricated
    # counterfactual pair. This ONE-scene diagnostic cannot pass the full
    # paired-source readiness check, which correctly requires both variants.
    meta.update(file=filename, candidate_only=True, variant=0, canonical_source=str(source),
        canonical_source_index_sha256=sha256_file(source/"index.json"), paired_scene_source=args.episode,
        paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
        recovery_start_actions_not_training_labels=[], original_payload_sha256=allowed[args.episode]["sha256"])
    meta["expert_terminal_requires_positive_contact"] = args.require_positive_contact
    meta["teacher_policy_source_sha256"] = sha256_file(Path(__file__).with_name("robust_touch_teacher.py"))
    meta["collector_source_sha256"] = sha256_file(Path(__file__))
    if args.require_positive_contact:
        meta["expert_terminal_force_quality_cap_n"] = 5.
        meta["expert_required_positive_contact_ticks"] = 2
    (args.output/"index.json").write_text(json.dumps([meta], indent=2)+"\n")
    manifest = copy.deepcopy(json.loads((source/"dataset_manifest.json").read_text()))
    for key in ("merge_components", "comparison_validation", "collection_plan_sha256", "independent_layouts",
                "split", "training_qualification"):
        manifest.pop(key, None)
    manifest.update(candidate_only=True, training_certified=False, stratified_target_vocabulary=False,
        episodes=1, frames=meta["steps"], declared_task_counts={"touch": 1}, episodes_per_task=None,
        purpose="Same-scene gentle touch contact-margin proposal; not qualified labels or model evidence",
        counterfactual_pair_complete=False,
        index_sha256=sha256_file(args.output/"index.json"))
    (args.output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"success": meta["success"], "steps": meta["steps"], "safety": meta["safety_violation"],
                      "original_reset_pixels_and_feedback_exact": True}), flush=True)
    if not meta["success"] or meta["safety_violation"]:
        raise RuntimeError("Preserved unsuccessful correction, do not train")


if __name__ == "__main__":
    main()
