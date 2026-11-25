"""Read-only touch-label physics diagnostics; NEVER a training qualification.

Inspect all touch episodes, preserving the original first-hit reward predicate.
Contact force is sampled at control boundaries, NOT every physics substep.
No scene filtering, relabelling, corrective data, or policy fallback is used.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def contact_classification(strict_success, surface_gap, maximum_force):
    if not strict_success:
        return "strict_failure"
    if maximum_force > 1e-6:
        return "positive_force_at_first_hit"
    if surface_gap is not None and surface_gap <= 0:
        return "nonpositive_gap_without_positive_force"
    return "positive_gap_or_no_surface_without_positive_force"


def replay(job):
    source, item, mode = job
    source = Path(source)
    path = source/item["file"]
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        actions = data["canonical_actions"]
        targets = data["next_joint_targets"]
        closures = native_gripper_closure(data)
        with audit_environment(source, image_size=32, required_success_steps=1,
                               use_semantic_success=False, render_observations=False) as env:
            reset_source_episode(env, source, item)
            initial_success = env.is_strict_success()
            max_sampled_force = 0.
            final_info = None
            final_gap = None
            final_force = 0.
            final_contact_count = 0
            for t, action in enumerate(actions):
                if mode == "native":
                    action = env.joint_targets_to_canonical(targets[t], float(closures[t]))
                _, _, done, final_info = env.step(action)
                contacts = p.getContactPoints(env.gripper.body, env.pick_id)
                closest = p.getClosestPoints(env.gripper.body, env.pick_id, distance=.02)
                final_gap = min((c[8] for c in closest), default=None)
                final_force = max((c[9] for c in contacts), default=0.)
                final_contact_count = len(contacts)
                max_sampled_force = max(max_sampled_force, final_force)
                if done:
                    break
            if final_info is None:
                raise ValueError("Empty episode")
            result = dict(file=item["file"], pick_name=item["task"]["pick_name"], mode=mode,
                initial_strict_success=initial_success, steps=t+1,
                strict_success=bool(final_info["strict_raw_success"]),
                terminal_surface_gap_m=final_gap, terminal_max_normal_force_n=final_force,
                terminal_contact_records=final_contact_count,
                maximum_control_boundary_force_n=max_sampled_force,
                safety_violation=final_info.get("safety_violation"), payload_sha256=digest)
            result["classification"] = contact_classification(result["strict_success"], final_gap, final_force)
    if sha256_file(path) != digest:
        raise RuntimeError("Source payload changed during read-only replay")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("both", "canonical", "native"), default="both")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    source = args.source.resolve()
    index_digest = sha256_file(source/"index.json")
    manifest_digest = sha256_file(source/"dataset_manifest.json")
    profile = json.loads((source/"dataset_manifest.json").read_text()).get("observation_profile")
    runtime = environment_runtime_fingerprint(profile)
    items = [r for r in json.loads((source/"index.json").read_text()) if r["task"]["name"] == "touch"]
    if not items:
        raise ValueError("No touch episodes")
    modes = ("canonical", "native") if args.mode == "both" else (args.mode,)
    jobs = [(str(source), item, mode) for mode in modes for item in items]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(replay, jobs):
            reports.append(result)
            print(json.dumps({"completed":len(reports), "total":len(jobs), **result}), flush=True)
    if (sha256_file(source/"index.json") != index_digest
            or sha256_file(source/"dataset_manifest.json") != manifest_digest
            or environment_runtime_fingerprint(profile) != runtime):
        raise RuntimeError("Read-only audit source/runtime changed")
    summary = {mode: dict(Counter(r["classification"] for r in reports if r["mode"] == mode)) for mode in modes}
    report = dict(scope="ALL touch episodes, original first-hit strict scoring; diagnostic only",
        source=str(source), index_sha256=index_digest, manifest_sha256=manifest_digest,
        environment_runtime_fingerprint=runtime, touch_episodes=len(items), modes=list(modes),
        control_boundary_sampling_only=True, training_qualified=False,
        validation_relabelled=False, summary=summary, reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
