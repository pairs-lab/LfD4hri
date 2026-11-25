"""Exact live-prefix transport diagnosis and separate offline repair candidates.

The source collector is never edited/stopped here. A saved visibility report
predeclares ALL failing rows; live progress fields may change but selected
payloads/rows/runtime/cameras/teacher implementation must remain identical.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from VLA.audit_acquisition_feedback_support import stable_manifest
from VLA.audit_native_camera_tensors import transformer
from VLA.audit_transport_visibility import transport_visibility
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility-report", type=Path, required=True)
    parser.add_argument("--teacher", help="module:Class; omit for original recorded-action replay")
    parser.add_argument("--checkpoint", required=True, help="Image config only; no neural weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    proof = json.loads(args.visibility_report.read_text())
    source = Path(proof["source"])
    snapshot = args.visibility_report.with_name(args.visibility_report.stem + "_index_snapshot.json")
    if sha256_file(snapshot) != proof["source_index_sha256"]:
        raise ValueError("Visibility report snapshot changed")
    original_index = {row["file"]: row for row in json.loads(snapshot.read_text())}
    selected = [row for row in proof["rows"] if row["failures"]]
    if not selected or any(row["task"]["name"] not in ("move", "pick_place") for row in selected):
        raise ValueError("Requires every reported move/place visibility failure")
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    runtime = manifest["environment_runtime_fingerprint"]["sources_sha256"]
    frozen = {str((source / row["file"]).resolve()): row["payload_sha256"] for row in selected}
    frozen.update({str(Path(path).resolve()): digest for path, digest in runtime.items()})
    for path in (Path(__file__), args.visibility_report, snapshot):
        frozen[str(path.resolve())] = sha256_file(path)
    cls = None
    if args.teacher:
        module, name = args.teacher.split(":")
        imported = importlib.import_module(module)
        cls = getattr(imported, name)
        frozen[str(Path(imported.__file__).resolve())] = sha256_file(Path(imported.__file__))
    def verify():
        latest = {row["file"]: row for row in json.loads((source / "index.json").read_text())}
        if any(latest.get(row["file"]) != original_index[row["file"]] for row in selected):
            raise ValueError("Selected committed row changed")
        current = json.loads((source / "dataset_manifest.json").read_text())
        if stable_manifest(manifest, True) != stable_manifest(current, True):
            raise ValueError("Source camera/robot/physics/collection contract changed")
        if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
            raise ValueError("Frozen source/runtime/proof/teacher changed")
    verify()
    args.output.mkdir(parents=True)
    plan = {"scope": "ALL committed transport-visibility failures; OFFLINE_ONLY_NOT_VLA_SR",
        "source": str(source), "visibility_report": str(args.visibility_report.resolve()),
        "selected_files": [row["file"] for row in selected], "teacher": args.teacher,
        "frozen_files": frozen, "runtime_changed": False, "cameras_changed": False,
        "training_performed": False, "training_authorized": False, "final_test_used": False}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    render, method = transformer(SimpleNamespace(model="smolvla", checkpoint=args.checkpoint))
    collected, reports, panels = [], [], []
    for selected_row in selected:
        verify()
        row = original_index[selected_row["file"]]
        with np.load(source / row["file"], allow_pickle=False) as source_arrays:
            references = {key: source_arrays[key] for key in source_arrays.files}
        with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
            from VLA.collect_expanded_v9_candidate import FrozenRewardReset
            from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset
            from VLA.common.success_contract_snapshot import recorded_distance_thresholds
            base = FrozenRewardReset(env, {row["task"]["name"]: recorded_distance_thresholds(row)})
            resetter = ClosedHoverRecoveryReset(base) if row.get("recovery_start_actions_not_training_labels") else base
            observation = resetter.reset(TaskSpec.from_dict(row["task"]))
            reset_errors = {key: float(np.max(np.abs(references[key][0].astype(float) - np.asarray(actual, float))))
                for key, actual in {"front_rgb": observation.front_rgb, "wrist_rgb": observation.wrist_rgb,
                    "joint_positions": observation.joint_positions, "gripper_closure": env.normalized_gripper_closure()}.items()}
            if any(reset_errors.values()):
                raise ValueError("Pinned exact reset mismatch: " + str(reset_errors))
            recorder = VideoRecorder(args.output / (Path(row["file"]).stem + ".mp4"), fps=10,
                overview_render=lambda: env.render("overview")) if args.video else None
            class Proxy:
                def __getattr__(self, name): return getattr(env, name)
                def reset(self, task): return observation
                def step(self, action):
                    if recorder: recorder.add(env.observe(), "OFFLINE path audit | " + row["task"]["prompt"])
                    return env.step(action)
            if cls:
                arrays, meta = collect_episode(Proxy(), TaskSpec.from_dict(row["task"]), policy_factory=cls)
            else:
                info = {}
                for action in references["canonical_actions"]:
                    _, _, done, info = Proxy().step(action)
                    if done: break
                arrays = references
                meta = {"success": bool(info.get("success")), "steps": env.step_count,
                    "safety_violation": info.get("safety_violation")}
            if recorder:
                recorder.add(env.observe(), "OFFLINE ONLY | success=" + str(meta["success"]))
                recorder.close()
        visibility, failures = transport_visibility(arrays, row["task"], max_transient_frames=1)
        from VLA.audit_preserved_touch_failure import preserved_reset_metadata
        meta.update(preserved_reset_metadata(row))
        meta.update(file=row["file"], candidate_only=True, task=row["task"],
                    scene_group=row["scene_group"], transport_repair_origin={"source": str(source),
                        "file": row["file"], "sha256": selected_row["payload_sha256"]})
        np.savez_compressed(args.output / row["file"], **arrays)
        collected.append(meta)
        reports.append({"file": row["file"], "reset_errors": reset_errors, "metadata": meta,
            "visibility": visibility, "visibility_failures": failures,
            "passed": bool(meta["success"] and not meta.get("safety_violation") and not failures)})
        # Inspect ALL original failing phases, with neighbouring frames; for
        # repaired candidates also inspect carry midpoint and termination.
        frames = sorted(set(min(len(arrays["canonical_actions"])-1, max(0, int(frame))) for frame in
            [*selected_row["roles"]["place"]["invisible_xy_transport_frames"], len(arrays["canonical_actions"])-1]))
        for frame in frames:
            panel = Image.new("RGB", (1024, 540), "white")
            ImageDraw.Draw(panel).text((5, 5), row["file"] + f" t={frame} goal=" + row["task"]["place_name"], fill="black")
            for column, view in enumerate(("front", "wrist")):
                _, native, mask = render(arrays[view+"_rgb"][frame], arrays[view+"_place_mask"][frame])[0]
                Image.fromarray(native).save(args.output / f"{Path(row['file']).stem}_t{frame:03d}_{view}.png")
                panel.paste(Image.fromarray(native), (column*512, 26))
            panels.append(panel)
        print(row["file"], "physical success", meta["success"], "visibility failures", failures, flush=True)
    verify()
    (args.output / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
    candidate_manifest = copy.deepcopy(manifest)
    for key in ("collection_plan_sha256", "split", "training_qualification"):
        candidate_manifest.pop(key, None)
    from collections import Counter
    candidate_manifest.update(candidate_only=True, training_certified=False, episodes=len(collected),
        frames=sum(row["steps"] for row in collected), declared_task_counts=dict(Counter(row["task"]["name"] for row in collected)),
        index_sha256=sha256_file(args.output / "index.json"))
    (args.output / "dataset_manifest.json").write_text(json.dumps(candidate_manifest, indent=2) + "\n")
    for number, panel in enumerate(panels):
        panel.save(args.output / f"native_visibility_panel{number:02d}.png")
    (args.output / "results.json").write_text(json.dumps({**plan, "model_image_method": method,
        "passed": all(row["passed"] for row in reports), "rows": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
