"""Fail-closed promotion of the opt-in numeric recipe after actual smoke.

Two optimizer updates, lossless reload, tensor parity and all-task deployed
integration are required. This qualifies numeric integration, NOT success.
"""
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.rdt.signal_precision import PROFILE


def require_signal_qualification(args):
    if getattr(args, "signal_precision", "bf16") == "bf16" or args.steps <= 2:
        return None
    path = getattr(args, "signal_qualification", None)
    if path is None:
        raise ValueError("FP32 signal long fit requires explicit successful GPU/integration qualification")
    report = json.loads(Path(path).read_text())
    if (report.get("passed") is not True or report.get("scope") != "RDT_FP32_signal_two_update_integration_qualification_NOT_task_success"
            or report.get("signal_precision") != PROFILE or report.get("neural_success_claimed") is not False):
        raise ValueError("Wrong/failed signal qualification")
    for filename, digest in report["bound_files"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError("Stale qualification file: " + filename)
    source = args.source.resolve()
    if report["source_index_sha256"] != sha256_file(source / "index.json") or report["source_manifest_sha256"] != sha256_file(source / "dataset_manifest.json"):
        raise ValueError("Signal qualification belongs to another source")
    cp = Path(report["smoke_checkpoint"])
    identity = verify_fingerprint(cp)
    if identity["content_tree_sha256"] != report["checkpoint_content_sha256"]:
        raise ValueError("Smoke checkpoint changed")
    manifest = json.loads((cp / "vla_checkpoint_manifest.json").read_text())
    for key in ("action_horizon", "episode_batch_size", "samples_per_episode", "gradient_accumulation", "learning_rate", "weight_decay", "optimizer", "seed", "signal_precision"):
        if manifest[key] != getattr(args, key):
            raise ValueError("Long fit differs from smoke recipe: " + key)
    for key, value in (("base_checkpoint", args.base_checkpoint), ("vision_checkpoint", args.vision_checkpoint), ("language_embeddings", args.language_embeddings)):
        if Path(manifest[key]).resolve() != value.resolve():
            raise ValueError("Long fit differs from smoke inputs: " + key)
    return report


def build_signal_qualification(source, checkpoint, old_audit, integration, output, guard_replay=None):
    source, cp, old, integration, output = map(lambda p: Path(p).resolve(), (source, checkpoint, old_audit, integration, output))
    if output.exists():
        raise FileExistsError("Preserve previous qualification")
    names = ("smoke_master_adam.json", "smoke_reload_precision.json", "qualified_tensor_parity.json", "input_smoke.json")
    reports = {name: json.loads((old / name).read_text()) for name in names}
    if any(r.get("passed") is not True or r.get("failures") for r in reports.values()):
        raise ValueError("Smoke/parity did not pass")
    master, reload = reports[names[0]], reports[names[1]]
    if master["real_optimizer_updates"] != 2 or master["optimizer_steps"] != [2.] or master["master_dtype"] != "float32" or master["adam_dtype"] != "float32":
        raise ValueError("Actual two-update FP32 master/Adam evidence required")
    if reload["lossless_loader_mismatched_values"] != 0 or reload["compared_float_values"] != master["master_weight_values"]:
        raise ValueError("Reload changed masters")
    for r in (master, reload):
        if Path(r["checkpoint"]).resolve() != cp:
            raise ValueError("Smoke reports bind another checkpoint")
    manifest = json.loads((cp / "vla_checkpoint_manifest.json").read_text())
    if manifest["steps"] != 2 or manifest["signal_precision"] != PROFILE:
        raise ValueError("Wrong saved smoke recipe")
    for key, file in (("source_index_sha256", "index.json"), ("source_manifest_sha256", "dataset_manifest.json")):
        if manifest["lineage"][key] != sha256_file(source / file):
            raise ValueError("Smoke/source mismatch")
    rollout = json.loads(integration.read_text())
    identity = verify_fingerprint(cp)
    if (rollout.get("scope") != "RDT_bounded_pipeline_integration_ONLY_NOT_success_rate"
            or rollout.get("trials") != 4 or rollout.get("integration_step_cap") != 2
            or rollout.get("initial_false_positives") != 0
            or rollout.get("checkpoint_content_sha256") != identity["content_tree_sha256"]
            or rollout.get("source_index_sha256") != sha256_file(source / "index.json")
            or rollout.get("oracle_fallback") is not False or rollout.get("privileged_information_in_controls") is not False
            or {r["task"] for r in rollout["results"]} != {"touch", "pick", "move", "pick_place"}
            or any(r["steps"] != 2 or len(r["trace"]) != 2 for r in rollout["results"])):
        raise ValueError("Four-task guarded deployed integration did not pass")
    # Numeric/API qualification is not a task-safety/performance claim for a
    # two-update policy. Never discard or reroll its unsafe command. Permit a
    # preserved guard event only after ALL recorded commands reproduce and
    # the rollback is independently verified byte-exact in the simulator.
    if rollout["safety_aborts"]:
        if guard_replay is None:
            raise ValueError("Smoke guard event needs independent exact replay/rollback proof")
        guard_replay = Path(guard_replay).resolve()
        guard = json.loads(guard_replay.read_text())
        if (guard.get("passed") is not True or guard.get("safety_guards_changed") is not False
                or guard.get("new_neural_trials") != 0
                or guard.get("bound_files", {}).get(str(integration)) != sha256_file(integration)
                or len(guard["rows"]) != 4):
            raise ValueError("Wrong/failed smoke guard replay")
        for filename, digest in guard["bound_files"].items():
            if sha256_file(Path(filename)) != digest: raise ValueError("Stale guard proof")
        reference = {r["file"]: r for r in rollout["results"]}
        for row in guard["rows"]:
            if (not row["passed"] or row["error"] or row["steps"] != 2
                    or row["reproduced_safety_abort"] != reference[row["file"]]["safety_abort"]):
                raise ValueError("Guard replay does not cover all original commands")
            if row["reproduced_safety_abort"] and (row.get("rollback_restored") is not True
                    or row["restored_state_max_error"] != 0 or row["pre_rollback_joint_error_rad"] != 0
                    or row["pre_rollback_velocity_error"] != 0):
                raise ValueError("Guard rollback is not exact")
    metadata = json.loads(integration.with_name("server_metadata.json").read_text())
    if metadata.get("signal_precision") != PROFILE or metadata.get("weight_dtype") != "float32" or metadata.get("compute") != "bfloat16_autocast":
        raise ValueError("Actual GPU server numeric profile differs")
    import VLA.rdt.server as server
    import VLA.rdt.checkpoint_loading as loader
    if reload["server_sha256"] != sha256_file(Path(server.__file__)) or reload["loader_sha256"] != sha256_file(Path(loader.__file__)):
        raise ValueError("Reload implementation changed")
    files = [old / name for name in names] + [integration, integration.with_name("server_metadata.json"),
        source / "index.json", source / "dataset_manifest.json", cp / "vla_checkpoint_manifest.json",
        Path(__file__), Path(server.__file__), Path(loader.__file__), Path(__file__).with_name("signal_precision.py")]
    if guard_replay is not None: files.append(Path(guard_replay))
    report = {"passed": True, "scope": "RDT_FP32_signal_two_update_integration_qualification_NOT_task_success",
        "signal_precision": PROFILE, "smoke_checkpoint": str(cp), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "source_index_sha256": sha256_file(source / "index.json"), "source_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
        "bound_files": {str(p): sha256_file(p) for p in files}, "neural_success_claimed": False,
        "smoke_task_safety_aborts_preserved": rollout["safety_aborts"], "safe_manipulation_established": False,
        "meaning": "Permits same numeric recipe/source bounded fitting after actual smoke and guarded deployment integration; not adequate learning or stable robot success"}
    output.write_text(json.dumps(report, indent=2)+"\n")
    return report
