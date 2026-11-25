"""Explicit RDT-specific bounded qualification; never promote an old recipe.

Inherits a verified SmolVLA common physics/split qualification ONLY as data
evidence, then adds current-source RDT FK, actual tensor parity, language and
actual native-image review. Creates a new immutable view, capped at 5k updates.
Neither learned success nor data sufficiency is asserted.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from VLA.common.provenance import sha256_file

TASKS = {"touch", "pick", "move", "pick_place"}
ROLE = "bounded_rdt_refinement_preconditions"
SCOPE = "bounded_rdt_refinement_experiment"


def check_inherited_budget(parent_limit, rdt_limit):
    """A model-specific child may narrow, never enlarge, the parent budget."""
    if (type(parent_limit) is not int or not 1001 <= parent_limit <= 20000
            or type(rdt_limit) is not int or not 1 <= rdt_limit <= min(5000, parent_limit)):
        raise ValueError("RDT budget must remain within its5k ceiling and the verified common parent cap")


def check_model_proofs(source, fk, parity, native, receipt):
    source = Path(source).resolve()
    index_hash = sha256_file(source / "index.json")
    documents = [json.loads(Path(path).read_text()) for path in (fk, parity, native, receipt)]
    action, tensors, images, reviewed = documents
    for report in (action, tensors, images):
        if Path(report.get("source", "")).resolve() != source or report.get("source_index_sha256") != index_hash:
            raise ValueError("RDT model proof is stale or for a different source")
    if (not action.get("passed") or action.get("source_manifest_sha256") != sha256_file(source / "dataset_manifest.json")
            or action.get("training_contract_sha256") != sha256_file(Path(__file__).with_name("rdt") / "training_data.py")):
        raise ValueError("RDT full native FK contract failed or changed")
    from VLA.common.runtime_provenance import verify_source_environment_runtime
    verify_source_environment_runtime(action, source)
    if (not tensors.get("passed") or tensors.get("samples", 0) < 128
            or set(tensors.get("errors", {})) != {"images", "states", "state_elem_mask", "lang_tokens", "lang_attn_mask", "ctrl_freqs"}
            or any(value != 0 for value in tensors["errors"].values())):
        raise ValueError("RDT actual train/inference parity failed")
    for name in ("server", "adapter", "training_data"):
        if tensors.get(name + "_sha256") != sha256_file(Path(__file__).with_name("rdt") / (name + ".py")):
            raise ValueError("RDT prepared tensor implementation changed")
    if images.get("model") != "rdt" or images.get("neural_weights_loaded") is not False:
        raise ValueError("Requires RDT's own image-preprocessor evidence")
    if (reviewed.get("model") != "rdt" or reviewed.get("source_index_sha256") != index_hash
            or reviewed.get("native_report_sha256") != sha256_file(native)
            or set(reviewed.get("tasks_reviewed", [])) != TASKS
            or reviewed.get("acceptable_for_bounded_fit") is not True
            or reviewed.get("neural_success_claimed") is not False
            or reviewed.get("data_sufficiency_established") is not False
            or len(reviewed.get("images", [])) < 5):
        raise ValueError("Missing actual four-task RDT native-image review")
    for image in reviewed["images"]:
        if sha256_file(image["path"]) != image["sha256"]:
            raise ValueError("Actually reviewed RDT image changed")
    expected = {(task, phase, camera) for task in TASKS for phase in ("reset", "middle", "final_expert")
                for camera in ("front", "wrist")}
    present = {(row["task"], row["phase"], row["camera"]) for row in images.get("samples", []) if row.get("finite")}
    if not expected <= present:
        raise ValueError("Missing finite RDT native images for a task/phase/view")
    return index_hash


def validate_precondition(document, index_hash, limit):
    if (document.get("passed") is not True or document.get("model") != "rdt"
            or document.get("scope") != SCOPE or document.get("source_index_sha256") != index_hash
            or document.get("max_optimizer_updates") != limit
            or document.get("data_sufficiency_established") is not False
            or document.get("final_test_used") is not False
            or document.get("inherited_common_physics_split_qualification") is not True):
        raise RuntimeError("RDT model-specific preconditions/budget changed")
    dependencies = document.get("model_proofs", [])
    if {row.get("role") for row in dependencies} != {"rdt_full_fk", "rdt_actual_tensor_parity", "rdt_native_images", "rdt_actual_visual_review", "rdt_language_manifest", "rdt_language_embeddings"}:
        raise RuntimeError("Incomplete RDT model-specific qualification proofs")
    for row in dependencies:
        if sha256_file(row["path"]) != row["sha256"]:
            raise RuntimeError("RDT model-specific qualification dependency changed")


def qualify(args):
    from VLA.common.source_training_gate import require_training_source
    from VLA.evaluate_smolvla_v9_source import verify_source_audit
    from VLA.rdt.training_data import validate_language_embeddings

    source, output = args.source.resolve(), args.output.resolve()
    precondition_path = output.with_name(output.name + "_rdt_preconditions.json")
    if output.exists() or precondition_path.exists():
        raise FileExistsError("Use a fresh RDT qualification, never rewrite an existing source")
    if type(args.max_updates) is not int or not 1 <= args.max_updates <= 5000:
        raise ValueError("This RDT experiment is bounded to 1..5000 real optimizer updates")
    require_training_source(source, model="smolvla", optimizer_updates=1)
    audit, manifest = verify_source_audit(source, args.source_audit)
    if manifest["training_qualification"]["scope"] != "bounded_refinement_experiment" or manifest.get("split") != "train":
        raise ValueError("Requires the full common larger-data TRAIN qualification")
    parent_limit = manifest["training_qualification"]["max_optimizer_updates"]
    check_inherited_budget(parent_limit, args.max_updates)
    index_hash = check_model_proofs(source, args.fk, args.parity, args.native, args.receipt)
    index = json.loads((source / "index.json").read_text())
    language_manifest = args.embeddings.with_suffix(".manifest.json")
    _, language = validate_language_embeddings(args.embeddings, language_manifest, source / "index.json",
                                              [row["task"]["prompt"] for row in index])
    if (language.get("encoder_weight_precision") != "bfloat16" or language.get("load_in_4bit") is not False
            or language.get("load_in_8bit") is not False):
        raise ValueError("Requires official unquantized BF16 T5 input cache")
    model_paths = [(args.fk, "rdt_full_fk"), (args.parity, "rdt_actual_tensor_parity"),
                   (args.native, "rdt_native_images"), (args.receipt, "rdt_actual_visual_review"),
                   (language_manifest, "rdt_language_manifest"), (args.embeddings, "rdt_language_embeddings")]
    preconditions = {"passed": True, "model": "rdt", "scope": SCOPE, "source_index_sha256": index_hash,
        "max_optimizer_updates": args.max_updates, "inherited_common_physics_split_qualification": True,
        "common_refinement_max_optimizer_updates": parent_limit,
        "data_sufficiency_established": False, "final_test_used": False,
        "source_audit_sha256": sha256_file(args.source_audit),
        "model_proofs": [{"path": str(Path(path).resolve()), "sha256": sha256_file(path), "role": role}
                         for path, role in model_paths],
        "known_limits": [f"Same{len(index)} TRAIN demos; no claim of sufficient missed-grasp recovery or composition.",
                         "Frozen T5/SigLIP, target-domain action expert, short horizon and replan1 experiment.",
                         "SmolVLA's heldout0/8 is preserved and does not qualify learned RDT success."]}
    validate_precondition(preconditions, index_hash, args.max_updates)
    output.mkdir(parents=True)
    for row in index:
        os.link(source / row["file"], output / row["file"])
    (output / "index.json").write_bytes((source / "index.json").read_bytes())
    precondition_path.write_text(json.dumps(preconditions, indent=2) + "\n")
    qualified = copy.deepcopy(manifest)
    qualified.update(qualification_parent=str(source), qualification_parent_index_sha256=index_hash,
                     qualification_parent_manifest_sha256=sha256_file(source / "dataset_manifest.json"),
                     candidate_only=False, training_certified=False,
                     purpose="RDT-specific finite learning experiment, NOT stable learned success or final baseline")
    common_proofs = qualified["training_qualification"]["proofs"]
    qualified["training_qualification"] = {"scope": SCOPE, "allowed_models": ["rdt"],
        "max_optimizer_updates": args.max_updates, "data_sufficiency_established": False, "final_test_used": False,
        "proofs": [*common_proofs, {"path": str(precondition_path), "sha256": sha256_file(precondition_path), "role": ROLE}],
        "known_limits": preconditions["known_limits"]}
    (output / "dataset_manifest.json").write_text(json.dumps(qualified, indent=2) + "\n")
    require_training_source(output, model="rdt", optimizer_updates=args.max_updates)
    print(json.dumps({"output": str(output), "episodes": len(index), "scope": SCOPE, "max_updates": args.max_updates}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "fk", "parity", "native", "receipt", "embeddings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=5000)
    qualify(parser.parse_args())


if __name__ == "__main__":
    main()
