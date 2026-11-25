"""Finite automatic full-replay failure -> same-scene repair -> gated fit.

Keeps the original failed report and every scene. Never turns a failed report
into a pass, skips a scene, edits a live workflow, or supplies a visual receipt.
Fresh full-source/split/image/conversion/GPU gates still precede optimization.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import verify_source_payload_fingerprint


def failed_touch_files(report, rows):
    indexed = {r["file"]: r for r in rows}
    if (report.get("all_episodes") is not True or report.get("mode") != "both"
            or report.get("native_success_mode") != "strict"
            or len(report.get("rows", [])) != 2 * len(rows)):
        raise ValueError("A complete both-mode report is required")
    expected = {(filename, mode) for filename in indexed for mode in ("canonical", "native")}
    actual = [(r["file"], r["mode"]) for r in report["rows"]]
    if set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("Missing or duplicate physical replay evidence")
    failed = [r for r in report["rows"] if not r["success"]]
    if (not failed or len(failed) > 8 or any(r.get("initial_strict_success") for r in report["rows"])
            or any(r["mode"] != "native" or indexed[r["file"]]["task"]["name"] != "touch"
                   or indexed[r["file"]].get("paired_scene_source") for r in failed)):
        raise ValueError("Only up to eight ordinary native-touch failures are covered by this repair")
    return sorted({r["file"] for r in failed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workflow", "output", "candidate", "fit-output", "model-output", "preview-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repair", type=Path, action="append", default=[])
    parser.add_argument("--wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    if any(p.exists() for p in (args.output, args.candidate, args.fit_output, args.model_output)):
        raise FileExistsError("Use fresh evidence, candidate and fit outputs")
    if not 1 <= args.wait_seconds <= 86400:
        raise ValueError("Finite replay wait required")
    workflow = args.workflow.resolve()
    source = workflow / "combined_candidate"
    old_state = json.loads((workflow / "workflow_status.json").read_text())
    source_manifest = json.loads((source / "dataset_manifest.json").read_text())
    if not source_manifest.get("candidate_only") or source_manifest.get("final_test_used"):
        raise ValueError("Only a pre-fit candidate is eligible")
    frozen = {str(p.resolve()): sha256_file(p) for p in
              (Path(__file__), Path(__file__).with_name("recollect_native_touch_failure.py"),
               source / "index.json", source / "dataset_manifest.json")}
    rows = json.loads((source / "index.json").read_text())
    args.output.mkdir(parents=True)
    state = {"source": str(source), "candidate": str(args.candidate.resolve()),
             "fit_output": str(args.fit_output.resolve()), "run_training_requested": True,
             "original_failure_preserved": True, "episodes_removed": 0,
             "frozen_files": frozen, "error": None, "final_test_used": False}
    def save(status, error=None):
        state.update(status=status, error=error, updated_at_utc=datetime.now(timezone.utc).isoformat())
        (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in frozen.items()):
            raise ValueError("Original source or repair implementation changed")
    def run(name, command):
        verify()
        save(name)
        with (args.output / (name + ".log")).open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        save("waiting_for_complete_original_both_replay")
        report_path = workflow / "combined_full_physics_replay.json"
        deadline = time.monotonic() + args.wait_seconds
        while not report_path.exists():
            verify()
            if time.monotonic() >= deadline:
                raise TimeoutError("Complete original replay missing")
            time.sleep(20)
        report = json.loads(report_path.read_text())
        if (Path(report["source"]).resolve() != source
                or report["source_index_sha256"] != sha256_file(source / "index.json")
                or report["source_manifest_sha256"] != sha256_file(source / "dataset_manifest.json")):
            raise ValueError("Replay refers to changed data")
        verify_source_payload_fingerprint(report["source_payload_fingerprint"], source)
        selected = failed_touch_files(report, rows)
        replacements = {}
        proof_paths = []
        existing = {}
        for path in args.repair:
            evidence = json.loads((path / "recollection.json").read_text())
            existing[evidence["original_episode"]] = path.resolve()
        for number, filename in enumerate(selected):
            repair = existing.get(filename, (args.output / ("repair_" + str(number))).resolve())
            if filename not in existing:
                run("collect_" + str(number), [sys.executable, "-m", "VLA.recollect_native_touch_failure",
                    "--source", str(source), "--episode", filename, "--output", str(repair)])
            evidence = json.loads((repair / "recollection.json").read_text())
            if (not evidence["passed"] or any(evidence["reset_errors"].values())
                    or evidence["runtime_changed"] or evidence["success_changed"]
                    or Path(evidence["source"]).resolve() != source):
                raise ValueError("Repair does not prove identical safe original reset")
            for p, digest in evidence["protected_files"].items():
                if sha256_file(Path(p)) != digest:
                    raise ValueError("Repair source/teacher changed")
            verify_source_payload_fingerprint(evidence["source_payload_fingerprint"], repair)
            replay_path = args.output / ("repair_" + str(number) + "_both_strict.json")
            run("replay_" + str(number), [sys.executable, "-m", "VLA.pi0.audit_closed_loop_replay",
                "--source", str(repair), "--output", str(replay_path), "--mode", "both",
                "--native-success-mode", "strict"])
            checked = json.loads(replay_path.read_text())
            if not checked["passed"] or checked["failures"]:
                raise ValueError("Repair still fails strict native/canonical execution")
            fresh, = json.loads((repair / "index.json").read_text())
            old = next(r for r in rows if r["file"] == filename)
            if (fresh["file"] != filename or fresh["task"] != old["task"]
                    or fresh["scene_group"] != old["scene_group"]
                    or fresh["native_touch_recollection_origin"]["sha256"] != sha256_file(source / filename)):
                raise ValueError("Repair changes exact scene/instruction")
            replacements[filename] = (repair, fresh)
            proof_paths += [repair / "recollection.json", replay_path]
        save("materializing_all_original_scenes_with_verified_replacements")
        args.candidate.mkdir(parents=True)
        materialized = []
        for row in rows:
            origin, fresh = replacements.get(row["file"], (source, row))
            os.link(origin / row["file"], args.candidate / row["file"])
            materialized.append(fresh)
        (args.candidate / "index.json").write_text(json.dumps(materialized, indent=2) + "\n")
        plan = {**state, "replaced_files": selected, "unchanged_payloads": len(rows) - len(selected),
                "original_replay": str(report_path), "original_replay_sha256": sha256_file(report_path),
                "proofs": {str(p.resolve()): sha256_file(p) for p in proof_paths},
                "training_authorized": False, "runtime_changed": False, "cameras_changed": False}
        (args.candidate / "repair_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        manifest = copy.deepcopy(source_manifest)
        for key in ("merge_components", "comparison_validation", "collection_plan_sha256"):
            manifest.pop(key, None)
        manifest.update(index_sha256=sha256_file(args.candidate / "index.json"),
                        frames=sum(r["steps"] for r in materialized),
                        repair_plan_sha256=sha256_file(args.candidate / "repair_plan.json"))
        (args.candidate / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.candidate / "collection_state.json").write_text(json.dumps({
            "status": "completed_candidate_not_qualified", "completed_episodes": len(rows),
            "planned_episodes": len(rows), "index_sha256": manifest["index_sha256"]}, indent=2) + "\n")
        counts = Counter(r["task"]["name"] for r in rows)
        merge_command = old_state["history"][0]["command"]
        comparison_validation = merge_command[merge_command.index("--validation") + 1]
        command = [sys.executable, "-m", "VLA.prepare_v9_candidate_fit",
            "--candidate", str(args.candidate), "--expected-task-counts", json.dumps(counts),
            "--comparison-validation", comparison_validation,
            "--output", str(args.fit_output), "--model-output", str(args.model_output),
            "--preview-checkpoint", str(args.preview_checkpoint), "--updates", str(old_state["updates"]),
            "--wait-seconds", str(args.wait_seconds), "--recovery-reset-observability", "dual_view",
            "--initial-frame-sampling-probability", str(old_state["initial_frame_sampling_probability"]),
            "--learning-rate", str(old_state["learning_rate"]), "--batch-size", str(old_state["batch_size"]),
            "--run-training"]
        run("running_fresh_full_gated_fit_workflow", command)
        save("completed_gated_fit_and_diagnostic_rollouts")
    except Exception as exc:
        save("stopped_no_silent_retry", str(exc))
        raise


if __name__ == "__main__":
    main()
