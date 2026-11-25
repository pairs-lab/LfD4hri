"""Fresh physical replay of every new privileged feature; no fake backfill.

Pilot8 must pass first. Original RGB/action labels are unchanged and checked
at every pre-action frame against existing numerical sensor tolerance. The
first18 fields must also agree with the already-qualified full state18.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit, reset_observation_errors, validate_reset_observation
from VLA.prepare_simstate_projection import select_pilot
from VLA.simstate_scene64 import SCHEMA, FEATURE_NAMES, scene_state, state_stats


def project(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError("Keep all previous projections/failures")
    _, manifest = verify_source_audit(source, args.source_audit)
    prior = json.loads((args.previous_projection / "audit.json").read_text())
    if (not prior.get("passed") or prior.get("pilot_only") or Path(prior["source"]).resolve() != source
            or prior["source_index_sha256"] != sha256_file(source / "index.json")
            or prior["source_manifest_sha256"] != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Require full qualified state18 projection of the same immutable source")
    rows = json.loads((source / "index.json").read_text()); selected = select_pilot(rows) if args.pilot else rows
    if not args.pilot:
        proof = json.loads(args.pilot_receipt.read_text())
        if (not proof.get("passed") or proof.get("pilot_only") is not True or proof.get("declared_episodes") != 8
                or proof.get("feature_builder_sha256") != sha256_file(Path(__file__).with_name("simstate_scene64.py"))
                or proof.get("source_index_sha256") != prior["source_index_sha256"]):
            raise ValueError("New native scene64 pilot8 must actually replay before full extraction")
    output.mkdir(parents=True); (output / "states").mkdir()
    plan = {"schema": SCHEMA, "feature_names": FEATURE_NAMES, "source": str(source),
        "source_index_sha256": prior["source_index_sha256"], "source_manifest_sha256": prior["source_manifest_sha256"],
        "source_audit": str(args.source_audit.resolve()), "source_audit_sha256": sha256_file(args.source_audit),
        "prior_projection_audit_sha256": sha256_file(args.previous_projection / "audit.json"),
        "declared_episodes": len(selected), "selected_files": [r["file"] for r in selected],
        "pilot_only": args.pilot, "split": "train", "candidate_only": True,
        "environment_runtime_fingerprint": environment_runtime_fingerprint(manifest["observation_profile"]),
        "extractor_sha256": sha256_file(Path(__file__)),
        "feature_builder_sha256": sha256_file(Path(__file__).with_name("simstate_scene64.py")),
        "teacher_action_or_phase_in_input": False, "future_state_in_input": False,
        "oracle_controller_at_deployment": False}
    atomic_json(output / "plan.json", plan); reports = []; arrays = []; frames = 0
    def save(status, error=None):
        atomic_json(output / "progress.json", {"status": status, "completed_episodes": len(reports),
            "planned_episodes": len(selected), "frames": frames, "rows": reports, "error": error,
            "neural_controls_executed": 0, "training_qualified": False})
    save("extracting_before_action_scene64")
    try:
        for row in selected:
            path = source / row["file"]; digest = sha256_file(path)
            old_path = args.previous_projection / "states" / row["file"]
            old_row = next(r for r in prior["rows"] if r["file"] == row["file"])
            if sha256_file(old_path) != old_row["state_payload_sha256"]: raise ValueError("Old causal state changed")
            with np.load(old_path, allow_pickle=False) as archive: old = archive["state"].copy()
            with np.load(path, allow_pickle=False) as data, audit_environment(source, image_size=384, control_hz=10,
                seed=0, required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
                obs = reset_source_episode(env, source, row); values = []; max_old = 0.; sensor_maxima = {}
                for frame, action in enumerate(data["canonical_actions"]):
                    errors = reset_observation_errors(obs, env.normalized_gripper_closure(),
                        {k: data[k][frame:frame+1] for k in ("joint_positions", "gripper_closure", "front_rgb", "wrist_rgb")})
                    validate_reset_observation(errors)
                    for key, error in errors.items(): sensor_maxima[key] = max(sensor_maxima.get(key, 0), error)
                    before = scene_state(env, obs); error = float(np.max(np.abs(before[:18] - old[frame])))
                    max_old = max(max_old, error)
                    if error > 1e-6: raise ValueError("Existing before-action18 timeline changed")
                    values.append(before)
                    if args.pilot and frame in (0, len(data["canonical_actions"]) // 2, len(data["canonical_actions"]) - 1):
                        folder = output / "visuals" / Path(row["file"]).stem; folder.mkdir(parents=True, exist_ok=True)
                        for camera in ("front", "wrist"): Image.fromarray(getattr(obs, camera + "_rgb")).save(folder / f"frame{frame:04d}_{camera}.png")
                        atomic_json(folder / f"frame{frame:04d}_scene.json", {"state": before.tolist(), "feature_names": FEATURE_NAMES})
                    obs, _, _, info = env.step(action)
                    if info.get("safety_violation"): raise ValueError("Replay safety failure")
                if not info["strict_raw_success"]: raise ValueError("Original strict terminal outcome no longer reached")
                states = np.asarray(values, np.float32); state_path = output / "states" / row["file"]
                np.savez_compressed(state_path, state=states); arrays.append(states); frames += len(states)
                reports.append({"file": row["file"], "task": row["task"]["name"], "frames": len(states),
                    "first18_max_error": max_old, "sensor_maxima": sensor_maxima, "terminal_strict_success": True,
                    "source_payload_sha256": digest, "state_payload_sha256": sha256_file(state_path)})
            if sha256_file(path) != digest: raise ValueError("Original source payload changed")
            save("extracting_before_action_scene64"); print(f"scene64 {len(reports)}/{len(selected)} episodes, {frames} frames", flush=True)
        if (sha256_file(source / "index.json") != plan["source_index_sha256"]
                or sha256_file(source / "dataset_manifest.json") != plan["source_manifest_sha256"]):
            raise ValueError("Original source changed")
        if not args.pilot and frames != prior["frames"]: raise ValueError("Incomplete full-source extraction")
        atomic_json(output / "stats.json", state_stats(np.concatenate(arrays)))
        atomic_json(output / "audit.json", {**plan, "passed": True, "frames": frames, "rows": reports,
            "stats_sha256": sha256_file(output / "stats.json"), "images_or_action_labels_modified": False,
            "qualified_for_training": False, "complete_causal_projection": not args.pilot,
            "own_model_tensor_CUDA_reload_qualification_still_required": True,
            "neural_success_established": False})
        save("pilot8_replayed_NOT_training_qualified" if args.pilot else "full64_extraction_passed_NOT_model_qualified")
    except Exception as error:
        save("failed_preserve_no_training", f"{type(error).__name__}: {error}"); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "previous-projection", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--pilot", action="store_true"); p.add_argument("--pilot-receipt", type=Path)
    args = p.parse_args()
    if not args.pilot and args.pilot_receipt is None: raise ValueError("Actual new-input pilot receipt required")
    project(args)


if __name__ == "__main__": main()
