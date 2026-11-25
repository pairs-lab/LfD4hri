"""Per-trajectory observability at the actual manipulation phases.

Global visibility averages can hide a fully occluded goal in a short carry.
Audit masks never enter a policy; raw384 visibility is an EXTRA gate, not a
replacement for official native-camera/preprocessing/loader audits.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.common.spatial import FIXED_DESTINATION


def transport_visibility(arrays, task, *, max_transient_frames=0, camera_views=("front", "wrist")):
    if max_transient_frames not in (0, 1):
        raise ValueError("Only the explicit zero/one-frame diagnostic protocols are supported")
    if tuple(camera_views) not in (("front",), ("front", "wrist")):
        raise ValueError("Use the native front-only or two-real-view diagnostic")
    actions = arrays["canonical_actions"]
    held_before = np.r_[False, arrays["next_grasped_object"][:-1].astype(bool)]
    verified_before = np.r_[False, arrays["next_verified_grasp"][:-1].astype(bool)]
    travel = held_before & (np.linalg.norm(actions[:, :2], axis=1) >= .005)
    result = {"held_xy_transport_frames": int(travel.sum()),
              "verified_xy_transport_frames": int((travel & verified_before).sum()), "roles": {}}
    failures = []
    for role in ("pick", "place"):
        applicable = role == "pick" or task.get("place_name") not in (None, *FIXED_DESTINATION)
        if not applicable:
            result["roles"][role] = {"physical_body": False, "note": "Named region has no physical mask; not an occlusion"}
            continue
        front = np.count_nonzero(arrays["front_"+role+"_mask"], axis=(1, 2))
        wrist = np.count_nonzero(arrays["wrist_"+role+"_mask"], axis=(1, 2))
        visible = (front >= 25) if tuple(camera_views) == ("front",) else (front >= 25) | (wrist >= 25)
        fraction = float(visible[travel].mean()) if travel.any() else None
        missing = int(np.count_nonzero(travel & ~visible))
        missing_indices = np.flatnonzero(travel & ~visible)
        runs = np.split(missing_indices, np.flatnonzero(np.diff(missing_indices) != 1) + 1)
        longest_gap = max((len(run) for run in runs), default=0)
        tolerated = bool(fraction is not None and fraction < .95 and fraction >= .80
                         and missing <= max_transient_frames and max_transient_frames == 1)
        result["roles"][role] = {"physical_body": True, "visible_xy_transport_fraction": fraction,
            "invisible_xy_transport_frames": np.flatnonzero(travel & ~visible).tolist(),
            "longest_consecutive_invisible_transport_frames": longest_gap,
            "visible_whole_episode_fraction": float(visible.mean()),
            "single_frame_outage_disclosed_not_full_observability": tolerated,
            "both_views_zero_pixels_frames": np.flatnonzero((front == 0) & (wrist == 0)).tolist()}
        if fraction is not None and fraction < .95 and not tolerated:
            failures.append(role+": per-trajectory transport visibility below95%")
        if max_transient_frames == 1 and longest_gap > 1:
            failures.append(role+": persistent transport occlusion exceeds explicit single-frame protocol")
    return result, failures


def audit(source, output, *, max_transient_frames=0, camera_views=("front", "wrist")):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    index_text = (source/"index.json").read_text()
    manifest_hash = sha256_file(source/"dataset_manifest.json")
    index = json.loads(index_text)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.stem+"_index_snapshot.json")
    if snapshot.exists():
        raise FileExistsError(snapshot)
    snapshot.write_text(index_text)
    state_path = source/"collection_state.json"
    complete = not state_path.exists() or json.loads(state_path.read_text())["status"].startswith("completed")
    rows, failures = [], []
    for item in index:
        path = source/item["file"]
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as arrays:
            result, errors = transport_visibility(arrays, item["task"], max_transient_frames=max_transient_frames,
                                                 camera_views=camera_views)
        if sha256_file(path) != digest:
            raise RuntimeError("Indexed payload changed during audit")
        rows.append({"file": item["file"], "task": item["task"], "payload_sha256": digest, **result, "failures": errors})
        failures.extend(item["file"]+": "+error for error in errors)
    if complete and (sha256_file(source/"index.json") != sha256_file(snapshot)
                     or sha256_file(source/"dataset_manifest.json") != manifest_hash):
        raise RuntimeError("Complete source changed during transport audit")
    report = {"scope": "per-trajectory raw-mask manipulation-phase gate; NOT native-model geometry or sufficient data",
        "source": str(source), "source_index_sha256": sha256_file(snapshot), "source_manifest_sha256": manifest_hash,
        "all_episodes": complete, "episodes": len(rows), "passed": not failures, "failures": failures,
        "protocol": "strict95pct_per_trajectory" if max_transient_frames == 0 else "bounded_fit_explicit_single100ms_outage_with_at_least80pct_transport_visibility",
        "full_observability_claimed": False,
        "camera_views_used_for_visibility": list(camera_views),
        "accepted_transient_outages": [{"file": row["file"], "role": role,
            "frames": value["invisible_xy_transport_frames"]} for row in rows for role, value in row["roles"].items()
            if value.get("single_frame_outage_disclosed_not_full_observability")],
        "policy_inputs_changed": False, "rows": rows}
    output.write_text(json.dumps(report, indent=2)+"\n")
    return {k:v for k,v in report.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-transient-frames", type=int, choices=(0, 1), default=0,
                        help="Explicit bounded-fit observability ablation, NOT a success-condition change; strict default remains unchanged")
    parser.add_argument("--camera-views", choices=("front", "front_wrist"), default="front_wrist",
                        help="Diagnostic of cameras the model ACTUALLY receives; two-view default qualification is unchanged")
    args = parser.parse_args()
    result = audit(args.source, args.output, max_transient_frames=args.max_transient_frames,
                   camera_views=("front",) if args.camera_views == "front" else ("front", "wrist"))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Preserved phase-visibility failures; do not hide them in global averages")


if __name__ == "__main__":
    main()
