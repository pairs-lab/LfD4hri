import json
from pathlib import Path
import tempfile
import unittest

from VLA.common.provenance import sha256_file
from VLA.common.source_training_gate import require_training_source
from VLA.qualify_rdt_v9_refinement import ROLE, SCOPE, validate_precondition, check_inherited_budget
from VLA.tests import test_source_training_gate as fixtures


class RDTRefinementGateTests(unittest.TestCase):
    def test_child_budget_can_narrow_a_larger_common_source(self):
        for parent, child in ((10000, 5000), (20000, 2), (5000, 5000), (2000, 1000)):
            check_inherited_budget(parent, child)
        for parent, child in ((10000, 5001), (2000, 3000), (1000, 500),
                              (True, 2), (10000, True), (10000, 0), (20001, 5000)):
            with self.assertRaises(ValueError): check_inherited_budget(parent, child)

    def bind_parent_budget(self, source, manifest, document, parent_limit):
        for proof in manifest["training_qualification"]["proofs"]:
            if proof.get("role") == "bounded_refinement_preconditions":
                path = Path(proof["path"])
                parent = json.loads(path.read_text())
                parent["max_optimizer_updates"] = parent_limit
                path.write_text(json.dumps(parent))
                proof["sha256"] = sha256_file(path)
            if proof.get("role") == ROLE:
                path = Path(proof["path"])
                document["common_refinement_max_optimizer_updates"] = parent_limit
                path.write_text(json.dumps(document))
                proof["sha256"] = sha256_file(path)
        (source / "dataset_manifest.json").write_text(json.dumps(manifest))

    def test_narrowed_child_verifies_original_parent_proof_and_keeps5k_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest, document = self.fixture(source)
            self.bind_parent_budget(source, manifest, document, 10000)
            require_training_source(source, model="rdt", optimizer_updates=5000)
            for model, updates in (("rdt", 5001), ("smolvla", 5000)):
                with self.assertRaises(RuntimeError):
                    require_training_source(source, model=model, optimizer_updates=updates)
            proof = next(p for p in manifest["training_qualification"]["proofs"] if p.get("role") == ROLE)
            document["common_refinement_max_optimizer_updates"] = 9999
            Path(proof["path"]).write_text(json.dumps(document))
            proof["sha256"] = sha256_file(proof["path"])
            (source / "dataset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "inherited common budget"):
                require_training_source(source, model="rdt", optimizer_updates=5000)

    def test_child_cannot_exceed_or_fabricate_a_lower_parent_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest, document = self.fixture(source)
            self.bind_parent_budget(source, manifest, document, 4000)
            with self.assertRaisesRegex(RuntimeError, "Refinement preconditions"):
                require_training_source(source, model="rdt", optimizer_updates=5000)

    def fixture(self, source):
        manifest = fixtures.SourceTrainingGateTest().refinement(source)
        paths = []
        for role in ("rdt_full_fk", "rdt_actual_tensor_parity", "rdt_native_images", "rdt_actual_visual_review", "rdt_language_manifest", "rdt_language_embeddings"):
            path = source / (role + ".json")
            path.write_text('{"passed": true}\n')
            paths.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
        document = {"passed": True, "model": "rdt", "scope": SCOPE,
                    "source_index_sha256": manifest["qualification_parent_index_sha256"],
                    "max_optimizer_updates": 5000, "data_sufficiency_established": False,
                    "final_test_used": False, "inherited_common_physics_split_qualification": True,
                    "model_proofs": paths}
        proof = source / "rdt_precondition.json"
        proof.write_text(json.dumps(document))
        qualification = manifest["training_qualification"]
        qualification.update(scope=SCOPE, allowed_models=["rdt"])
        qualification["proofs"].append({"path": str(proof), "sha256": sha256_file(proof), "role": ROLE})
        (source / "dataset_manifest.json").write_text(json.dumps(manifest))
        return manifest, document

    def test_rdt_has_its_own_scope_and_smoke_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.fixture(source)
            for updates in (2, 5000):
                require_training_source(source, model="rdt", optimizer_updates=updates)
            for model in ("smolvla", "openvla", "pi0"):
                with self.assertRaisesRegex(RuntimeError, "model is not qualified"):
                    require_training_source(source, model=model, optimizer_updates=2)

    def test_existing_smolvla_qualification_does_not_authorize_rdt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            fixtures.SourceTrainingGateTest().refinement(source)
            with self.assertRaises(RuntimeError):
                require_training_source(source, model="rdt", optimizer_updates=2)

    def test_changing_scope_only_cannot_authorize_rdt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest = fixtures.SourceTrainingGateTest().refinement(source)
            manifest["training_qualification"].update(scope=SCOPE, allowed_models=["rdt"])
            (source / "dataset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "RDT model-specific"):
                require_training_source(source, model="rdt", optimizer_updates=2)

    def test_changed_model_dependency_rejects_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            _, document = self.fixture(source)
            Path(document["model_proofs"][0]["path"]).write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "dependency changed"):
                require_training_source(source, model="rdt", optimizer_updates=2)

    def test_new_scope_cannot_raise_the_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest, _ = self.fixture(source)
            manifest["training_qualification"]["max_optimizer_updates"] = 5001
            (source / "dataset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "Invalid bounded"):
                require_training_source(source, model="rdt", optimizer_updates=5001)

    def test_bad_preconditions_reject_claims_or_missing_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            manifest, document = self.fixture(source)
            for key, value in (("final_test_used", True), ("data_sufficiency_established", True), ("model", "smolvla"), ("passed", False)):
                wrong = {**document, key: value}
                with self.assertRaises(RuntimeError):
                    validate_precondition(wrong, manifest["qualification_parent_index_sha256"], 5000)
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                validate_precondition({**document, "model_proofs": []}, manifest["qualification_parent_index_sha256"], 5000)


if __name__ == "__main__":
    unittest.main()
