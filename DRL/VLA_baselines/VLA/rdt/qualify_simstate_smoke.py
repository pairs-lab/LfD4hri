"""Bind actual privileged CUDA, exact tensors, reload and deployed controls.

Qualifies a bounded finetuning experiment, never stable/high task success.
"""
import argparse
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.rdt.simstate_inputs import SCHEMA
from VLA.rdt.signal_precision import PROFILE


def build(checkpoint, smoke, tensor_proof, output):
    if output.exists():
        raise FileExistsError("Preserve each qualification")
    paths = [smoke/"master_adam.json", smoke/"lossless_reload.json", tensor_proof,
             smoke/"deployed_four_task_smoke/analysis.json", smoke/"deployed_four_task_smoke/server_metadata.json"]
    master, reload, tensors, integration, metadata = [json.loads(p.read_text()) for p in paths]
    if (not master.get("passed") or master["real_optimizer_updates"] != 2 or master["optimizer_steps"] != [2.]
            or master["master_dtype"] != "float32" or master["adam_dtype"] != "float32"
            or not reload.get("passed") or reload["lossless_loader_mismatched_values"] != 0
            or reload["compared_float_values"] != master["master_weight_values"]
            or Path(master["checkpoint"]).resolve() != checkpoint.resolve()
            or Path(reload["checkpoint"]).resolve() != checkpoint.resolve()):
        raise ValueError("Actual privileged two-update/full-master reload proof failed")
    manifest = json.loads((checkpoint/"vla_checkpoint_manifest.json").read_text())
    if (manifest["steps"] != 2 or manifest["signal_precision"] != PROFILE
            or manifest.get("privileged_model_inputs") is not True or manifest.get("privileged_pose_consumed") is not True):
        raise ValueError("Wrong actual saved training recipe")
    source = Path(manifest["lineage"]["source"])
    if (not tensors.get("passed") or tensors["samples"] != 128 or tensors["schema"] != SCHEMA
            or any(tensors["errors"].values()) or tensors["native_action_indices_unchanged"] != [0,1,2,3,4,5,10]
            or tensors["source_index_sha256"] != sha256_file(source/"index.json")
            or tensors["projection_audit_sha256"] != manifest["simstate"]["projection_audit_sha256"]
            or tensors["input_implementation_sha256"] != sha256_file(Path(__file__).with_name("simstate_inputs.py"))):
        raise ValueError("Actual model-specific input parity failed or changed")
    identity = verify_fingerprint(checkpoint, full=True)
    if (integration.get("status") != "completed_four_task_API_smoke_NOT_SR" or integration["actual_controls"] != 8
            or not integration["all_four_tasks_reached"] or not integration["all_two_controls_completed"]
            or integration["checkpoint_content_sha256"] != identity["content_tree_sha256"]
            or integration.get("oracle_controller_at_deployment") is not False
            or integration.get("final_or_development_scenes_used") is not False
            or {r["task"] for r in integration["rows"]} != {"touch","pick","move","pick_place"}
            or any(r["steps"] != 2 or r["safety_abort"] for r in integration["rows"])
            or metadata.get("simstate_schema") != SCHEMA or metadata.get("privileged_model_inputs") is not True
            or metadata.get("signal_precision") != PROFILE or metadata.get("weight_dtype") != "float32"
            or metadata.get("compute") != "bfloat16_autocast"
            or metadata["checkpoint"]["content_tree_sha256"] != identity["content_tree_sha256"]):
        raise ValueError("New privileged four-task GPU/deployed integration failed")
    for row in integration["rows"]:
        for step in row["trace"]:
            if len(step["pre_action_privileged_state18"]) != 18 or len(step["action"]) != 7 or step["safety_violation"]:
                raise ValueError("Incomplete physical/control trace")
    implementations = [Path(__file__), Path(__file__).with_name("train_simstate.py"),
        Path(__file__).with_name("simstate_server.py"), Path(__file__).with_name("simstate_adapter.py"),
        Path(__file__).with_name("simstate_inputs.py"), Path(__file__).with_name("training.py"),
        Path(__file__).with_name("checkpoint_loading.py"), Path(__file__).with_name("signal_precision.py"),
        Path(__file__).with_name("server.py"), Path(__file__).resolve().parents[1]/"evaluate_rdt_simstate_smoke.py"]
    paths += implementations + [source/"index.json", source/"dataset_manifest.json",
        checkpoint/"vla_checkpoint_manifest.json", Path(manifest["simstate"]["projection"])/"audit.json"]
    bound = {str(p.resolve()): sha256_file(p) for p in paths}
    common = {"passed": True, "smoke_checkpoint": str(checkpoint.resolve()),
        "checkpoint_content_sha256": identity["content_tree_sha256"], "bound_files": bound,
        "signal_precision": PROFILE, "source_index_sha256": sha256_file(source/"index.json"),
        "source_manifest_sha256": sha256_file(source/"dataset_manifest.json"),
        "privileged_model_inputs": True, "neural_success_claimed": False,
        "safe_or_successful_full_task_execution_established": False}
    output.mkdir(parents=True)
    # This native numeric guard additionally verifies exact saved batch/LR/
    # optimizer/base recipe for the forthcoming long fit. Its scope does not
    # claim an unprivileged input experiment; the new model guard is separate.
    atomic_json(output/"native_signal_qualification.json", {**common,
        "scope": "RDT_FP32_signal_two_update_integration_qualification_NOT_task_success"})
    atomic_json(output/"simstate_qualification.json", {**common,
        "schema": SCHEMA, "scope": "RDT_SimState_actual_two_update_reload_four_task_integration_NOT_SR",
        "optimizer_steps": [2.], "native_action_indices_unchanged": [0,1,2,3,4,5,10],
        "projection_audit_sha256": manifest["simstate"]["projection_audit_sha256"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "smoke", "tensor-proof", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.checkpoint, args.smoke, args.tensor_proof, args.output)
