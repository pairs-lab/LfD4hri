"""Replay observed TRAIN-only grasp failures, then demonstrate real recovery.

This is privileged teacher/data evidence, NOT learned success. Bad learned
prefixes are setup controls, never expert labels or policy observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.collect_demos import collect_episode
from VLA.common.tasks import TaskSpec
from VLA.common.recorder import VideoRecorder
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.recovery_teacher import RecoveryTeacher


class FailedPrefixReset:
    def __init__(self, env, source, row, prefix):
        self.env, self.source, self.row, self.prefix = env, source, row, prefix
        self.recorder = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        if json.dumps(task.to_dict(), sort_keys=True) != json.dumps(self.row["task"], sort_keys=True):
            raise ValueError("Correction must use the identical source task")
        reset_source_episode(self.env, self.source, self.row)
        for action in self.prefix:
            _, _, done, info = self.env.step(np.asarray(action, dtype=np.float32))
            if done or info["grasped_object"] is not None:
                raise ValueError("The supposedly failed prefix completed/grasped; no empty-hand label allowed")
        if self.env.normalized_gripper_closure() < .5:
            raise ValueError("Prefix does not reproduce a closed-empty hand")
        self.env.step_count = self.env.success_streak = 0
        self.env._last_success = False
        return self.env.observe()

    def step(self, action):
        result = self.env.step(action)
        if self.recorder is not None:
            self.recorder.add(result[0], "Privileged teacher correction ONLY, not VLA success")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.source / "dataset_manifest.json").read_text())
    replay = json.loads(args.replay.read_text())
    if (not manifest.get("tiny_fit_only") or replay.get("final_test_used")
            or replay["source_index_sha256"] != sha256_file(args.source / "index.json")):
        raise ValueError("Requires immutable tiny TRAIN replay, not development/final failures")
    bad = next(r for r in replay["results"] if r["task"] == "pick" and not r["success"])
    row = next(r for r in json.loads((args.source / "index.json").read_text()) if r["file"] == bad["file"])
    first_close = next(i for i, a in enumerate(bad["actions"]) if a[6] > .2)
    args.output.mkdir(parents=True)
    records = []
    with audit_environment(args.source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.25, use_semantic_success=True) as env:
        for count in (first_close + 1, first_close + 7):
            prefix = bad["actions"][:count]
            proxy = FailedPrefixReset(env, args.source.resolve(), row, prefix)
            task = TaskSpec.from_dict(row["task"])
            start = proxy.reset(task)
            folder = args.output / f"prefix{count}"
            folder.mkdir()
            for camera in ("front", "wrist"):
                Image.fromarray(getattr(start, camera + "_rgb")).save(folder / f"start_{camera}.png")
            recorder = VideoRecorder(folder / "teacher_correction.mp4", fps=10,
                                     overview_render=lambda: env.render("overview"))
            proxy.recorder = recorder
            try:
                recorder.add(start, "Actual learned failed prefix -> privileged TEACHER correction")
                arrays, metadata = collect_episode(proxy, task, policy_factory=RecoveryTeacher)
            finally:
                recorder.close()
            if arrays["canonical_actions"][0, 6] >= 0:
                raise ValueError("Correction failed to reopen the empty hand")
            np.savez_compressed(folder / "correction.npz", **arrays)
            for camera in ("front", "wrist"):
                Image.fromarray(env.render(camera)).save(folder / f"terminal_{camera}.png")
            metadata.update(source_failed_episode=bad["file"], failed_prefix_length=count,
                recovery_start_actions_not_training_labels=prefix, scope="Teacher-only recovery, NOT learned performance")
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            records.append({"prefix_length": count, "closed_empty_start": True,
                "first_label_reopens": True, "teacher_success": metadata["success"], "steps": metadata["steps"]})
            print(json.dumps(records[-1]), flush=True)
    report = {"passed": all(r["teacher_success"] for r in records), "records": records,
        "scope": "Actual TRAIN failure-prefix replay and privileged teacher correction only",
        "source": str(args.source.resolve()), "replay_sha256": sha256_file(args.replay),
        "final_test_used": False, "trained_model_changed": False, "runtime_oracle_fallback": False}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("Teacher could not recover; do not qualify correction")


if __name__ == "__main__":
    main()
