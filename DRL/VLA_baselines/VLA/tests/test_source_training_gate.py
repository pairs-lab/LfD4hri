import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.source_training_gate import require_training_source
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint


class SourceTrainingGateTest(unittest.TestCase):
    def check(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "dataset_manifest.json").write_text(json.dumps(manifest))
            require_training_source(source)

    def test_legacy_sources_preserve_compatibility(self):
        self.check({"robot": "ur5e"})

    def test_explicit_qualified_derivative_is_allowed(self):
        self.check({"candidate_only": False})

    def test_unqualified_candidate_cannot_train(self):
        with self.assertRaisesRegex(RuntimeError, "Unqualified candidate"):
            self.check({"candidate_only": True})

    def test_v9_cannot_promote_without_binding_real_controller_code(self):
        with self.assertRaisesRegex(RuntimeError, "runtime fingerprint"):
            self.check({"candidate_only": False, "observation_profile": {"gripper_controller": "body_contact_v9"}})

    def qualified(self, source):
        profile = {"gripper_controller": "body_contact_v9"}
        (source / "index.json").write_text("[]\n")
        proof = source / "proof.json"
        proof.write_text('{"passed": true}\n')
        manifest = {"candidate_only": False, "observation_profile": profile,
                    "environment_runtime_fingerprint": environment_runtime_fingerprint(profile),
                    "qualification_parent_index_sha256": sha256_file(source / "index.json"),
                    "training_qualification": {"scope": "bounded_pipeline_pilot", "allowed_models": ["smolvla"],
                                               "max_optimizer_updates": 500,
                                               "proofs": [{"path": str(proof), "sha256": sha256_file(proof)}]}}
        (source / "dataset_manifest.json").write_text(json.dumps(manifest))
        return proof

    def test_bounded_v9_pilot_allows_only_declared_recipe_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.qualified(source)
            require_training_source(source, model="smolvla", optimizer_updates=500)
            for updates in (None, 501, 10000, 0, True):
                with self.assertRaisesRegex(RuntimeError, "at most"):
                    require_training_source(source, model="smolvla", optimizer_updates=updates)
            with self.assertRaisesRegex(RuntimeError, "model is not qualified"):
                require_training_source(source, model="openvla", optimizer_updates=500)

    def test_changed_qualification_proof_rejects_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            proof = self.qualified(source)
            proof.write_text('{"passed": false}\n')
            with self.assertRaisesRegex(RuntimeError, "proof changed"):
                require_training_source(source, model="smolvla", optimizer_updates=10)

    def refinement(self, source):
        self.qualified(source)
        path = source/"dataset_manifest.json"
        manifest = json.loads(path.read_text())
        precondition = source/"refinement.json"
        precondition.write_text(json.dumps({"passed": True, "max_optimizer_updates": 5000,
            "source_index_sha256": sha256_file(source/"index.json")}))
        qualification = manifest["training_qualification"]
        qualification.update(scope="bounded_refinement_experiment", max_optimizer_updates=5000)
        qualification["proofs"].append({"path": str(precondition), "sha256": sha256_file(precondition),
                                        "role": "bounded_refinement_preconditions"})
        path.write_text(json.dumps(manifest))
        return manifest

    def test_refinement_has_separate_budget_and_immutable_preconditions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest = self.refinement(source)
            require_training_source(source, model="smolvla", optimizer_updates=5000)
            require_training_source(source, model="smolvla", optimizer_updates=2)
            with self.assertRaises(RuntimeError):
                require_training_source(source, model="smolvla", optimizer_updates=5001)
            manifest["training_qualification"]["proofs"] = manifest["training_qualification"]["proofs"][:1]
            (source/"dataset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "preconditions"):
                require_training_source(source, model="smolvla", optimizer_updates=5000)

    def test_old_pilot_scope_cannot_inherit_refinement_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest = self.refinement(source)
            manifest["training_qualification"]["scope"] = "bounded_pipeline_pilot"
            (source/"dataset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "Invalid bounded"):
                require_training_source(source, model="smolvla", optimizer_updates=5000)
