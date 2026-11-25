"""Inspect actual bilateral grasp surface normals, not just held-body flags.

Deterministic first PICK per vocabulary name in a committed snapshot. Supports
live collection diagnosis; binds selected immutable payloads, NEVER qualifies
the full source or changes controllers or learned-policy inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    index_text = (args.source/"index.json").read_text()
    (args.output/"index_snapshot.json").write_text(index_text)
    selected = {}
    for row in json.loads(index_text):
        if row["task"]["name"] == "pick":
            selected.setdefault(row["task"]["pick_name"], row)
    if not selected:
        raise ValueError("No committed PICK episodes")
    reports = []
    with audit_environment(args.source, image_size=32, render_observations=False,
                           required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
        for name, item in selected.items():
            payload = args.source/item["file"]
            digest = sha256_file(payload)
            with np.load(payload, allow_pickle=False) as archive:
                actions = archive["canonical_actions"].copy()
            reset_source_episode(env, args.source, item)
            frames = []
            first_held = None
            first_bilateral_held = None
            for t, action in enumerate(actions):
                _, _, done, info = env.step(action)
                if action[6] > 0:
                    lo, hi = map(np.asarray, p.getAABB(env.pick_id))
                    a, b = env.gripper.pad_positions()
                    axis = np.asarray(b-a, float)
                    axis /= max(np.linalg.norm(axis), 1e-9)
                    contacts = [c for c in p.getContactPoints(bodyA=env.gripper.body, bodyB=env.pick_id)
                                if c[3] in env.gripper.PAD_LINKS and c[9] > .1]
                    evidence = [{"pad_link": c[3], "force_N": c[9], "normal_on_object": c[7],
                        "normal_jaw_projection": float(np.dot(c[7], axis)),
                        "contact_world": c[6], "contact_height_fraction": float((c[6][2]-lo[2])/max(hi[2]-lo[2], 1e-9)),
                        "side_facing": bool(abs(c[7][2]) <= .5 and abs(np.dot(c[7], axis)) >= .5)} for c in contacts]
                    sides = {c["pad_link"] for c in evidence if c["side_facing"]}
                    held = env.gripper.grasped_object == env.pick_id
                    bilateral = env.gripper._bilateral(env.pick_id)
                    frame = {"frame": t, "held_body_id_matches": held, "verified_held": held and bilateral, "bilateral": bilateral,
                        "two_pad_side_contacts": set(env.gripper.PAD_LINKS) <= sides,
                        "tcp_height_fraction": float((env.gripper.grasp_center()[2]-lo[2])/max(hi[2]-lo[2], 1e-9)),
                        "contacts": evidence}
                    if held and first_held is None:
                        first_held = frame
                    if held and bilateral and first_bilateral_held is None:
                        first_bilateral_held = frame
                    frames.append(frame)
                if done:
                    break
            if sha256_file(payload) != digest:
                raise RuntimeError("Selected immutable episode changed")
            report = {"name": name, "file": item["file"], "payload_sha256": digest,
                "terminal_success": info["success"], "safety_violation": info.get("safety_violation"),
                "first_held": first_held, "first_bilateral_held": first_bilateral_held, "close_frames": frames,
                "held_frames": sum(f["held_body_id_matches"] for f in frames),
                "verified_held_frames": sum(f["verified_held"] for f in frames),
                "held_frames_with_two_pad_side_contacts": sum(f["held_body_id_matches"] and f["two_pad_side_contacts"] for f in frames)}
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k not in ("close_frames", "first_held", "first_bilateral_held")}), flush=True)
    report = {"scope": "Selected committed expert-label physical surface audit, NOT full qualification or learned SR",
        "source": str(args.source.resolve()), "snapshot_index_sha256": sha256_file(args.output/"index_snapshot.json"),
        "selected_targets": len(reports), "selection": "first committed PICK per name, no outcome selection",
        "physics_or_policy_changed": False, "training_certified": False,
        "all_first_holds_have_two_pad_side_contacts": all(r["first_held"] and r["first_held"]["two_pad_side_contacts"] for r in reports),
        "all_first_bilateral_holds_have_two_pad_side_contacts": all(r["first_bilateral_held"] and r["first_bilateral_held"]["two_pad_side_contacts"] for r in reports),
        "episodes": reports}
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
