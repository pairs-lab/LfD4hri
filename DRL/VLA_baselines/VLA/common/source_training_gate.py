"""Prevent diagnostic candidate sources from silently becoming training data."""
import json
from pathlib import Path


def require_training_source(source, *, model=None, optimizer_updates=None):
    """Legacy sources are unchanged; opt-in candidates require promotion first.

    Conversion and tensor audits may still inspect candidates. Promotion must
    create a separately audited immutable source, not toggle an in-use file.
    """
    manifest_path = Path(source) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("candidate_only") is True:
        raise RuntimeError(
            "Unqualified candidate-only source: do not fine-tune before physics, "
            "camera, feedback/action and independent-scene audits pass; create "
            "an immutable qualified training view first"
        )
    profile = manifest.get("observation_profile") or {}
    if profile.get("gripper_controller") == "body_contact_v9":
        from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
        verify_environment_runtime_fingerprint(manifest.get("environment_runtime_fingerprint"), profile)
        qualification = manifest.get("training_qualification", {})
        scope = qualification.get("scope")
        if scope not in {"bounded_pipeline_pilot", "bounded_refinement_experiment", "bounded_rdt_refinement_experiment"}:
            raise RuntimeError("V9 training source lacks explicit bounded-pilot qualification")
        limit = qualification.get("max_optimizer_updates")
        ceiling = {"bounded_pipeline_pilot": 1000, "bounded_refinement_experiment": 20000,
                   "bounded_rdt_refinement_experiment": 5000}[scope]
        if type(limit) is not int or not 1 <= limit <= ceiling or not qualification.get("proofs"):
            raise RuntimeError("Invalid bounded-pilot qualification")
        if scope == "bounded_refinement_experiment":
            if (limit <= 1000 or qualification.get("allowed_models") != ["smolvla"]
                    or not any(proof.get("role") == "bounded_refinement_preconditions" for proof in qualification["proofs"])):
                raise RuntimeError("Missing explicit refinement budget/data/visual preconditions")
        if scope == "bounded_rdt_refinement_experiment":
            if (qualification.get("allowed_models") != ["rdt"]
                    or not any(proof.get("role") == "bounded_rdt_refinement_preconditions" for proof in qualification["proofs"])):
                raise RuntimeError("Missing explicit RDT model-specific preconditions")
        if model not in qualification.get("allowed_models", []):
            raise RuntimeError("This model is not qualified for the bounded source pilot")
        if type(optimizer_updates) is not int or not 1 <= optimizer_updates <= limit:
            raise RuntimeError(f"Source is qualified only for a bounded pilot of at most {limit} optimizer updates")
        from VLA.common.provenance import sha256_file
        common_limits = []
        for proof in qualification["proofs"]:
            if sha256_file(Path(proof["path"])) != proof["sha256"]:
                raise RuntimeError("Bounded-pilot qualification proof changed")
            if proof.get("role") == "bounded_refinement_preconditions":
                preconditions = json.loads(Path(proof["path"]).read_text())
                common_limit = preconditions.get("max_optimizer_updates")
                # Common physics/coverage proof remains SHA-bound. Only the
                # separately qualified RDT child may REDUCE the inherited cap.
                # SmolVLA's exact-budget requirement is unchanged.
                valid_budget = common_limit == limit
                if scope == "bounded_rdt_refinement_experiment":
                    valid_budget = type(common_limit) is int and max(1001, limit) <= common_limit <= 20000
                if (not preconditions.get("passed") or not valid_budget
                        or preconditions.get("source_index_sha256") != manifest.get("qualification_parent_index_sha256")):
                    raise RuntimeError("Refinement preconditions/budget changed")
                common_limits.append(common_limit)
            if proof.get("role") == "bounded_rdt_refinement_preconditions":
                from VLA.qualify_rdt_v9_refinement import validate_precondition
                validate_precondition(json.loads(Path(proof["path"]).read_text()),
                                      manifest.get("qualification_parent_index_sha256"), limit)
        if scope == "bounded_rdt_refinement_experiment":
            for proof in qualification["proofs"]:
                if proof.get("role") == "bounded_rdt_refinement_preconditions":
                    child = json.loads(Path(proof["path"]).read_text())
                    # Historical equal-cap documents remain valid. A narrowed
                    # new child must explicitly name the original proof cap.
                    inherited = child.get("common_refinement_max_optimizer_updates", limit)
                    if not common_limits or any(value != inherited for value in common_limits):
                        raise RuntimeError("RDT inherited common budget does not match its immutable parent proof")
        if sha256_file(Path(source) / "index.json") != manifest.get("qualification_parent_index_sha256"):
            raise RuntimeError("Qualified-pilot payload index changed")
