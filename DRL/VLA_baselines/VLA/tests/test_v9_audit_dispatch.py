"""Audit entry points must not silently replay V9 data with legacy physics."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import baseline_profile, profile_hash, ProfiledManipulationEnv


class DispatchReached(Exception):
    pass


class V9AuditDispatchTest(unittest.TestCase):
    MODULES = (
        "VLA.audit_counterfactual_collection",
        "VLA.audit_evaluation_protocol",
        "VLA.rerender_camera_candidate",
        "VLA.audit_destination_counterfactual",
        "VLA.audit_grasp_task_independence",
    )

    def fixture(self, root, v9=True):
        source = root / "source"
        source.mkdir()
        profile = baseline_profile()
        if v9:
            profile.update(gripper_controller="body_contact_v9",
                           grasp_candidate_scope="all_dynamic_scene_objects",
                           gripper_state_contract="encoder_fk_aperture_closure_v1")
        profile_path = root / "profile.json"
        profile_path.write_text(json.dumps(profile))
        (source / "dataset_manifest.json").write_text(json.dumps({"observation_profile": profile}))
        (source / "index.json").write_text("[]")
        (source / "audit.json").write_text(json.dumps({"passed": True, "source": str(source)}))
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({
            "observation_profile_file": str(profile_path),
            "observation_profile_sha256": profile_hash(profile),
            "development_seeds": [20262001], "final_test_seeds": [20263001],
            "train_dataset_seed": 20261010,
        }))
        args = SimpleNamespace(source=source, profile=profile_path, observation_profile=profile_path,
                               output=root / "output", layouts_per_task=1, resume=False,
                               native_command_labels=False, recovery_start_steps=0,
                               closed_empty_workspace_start=False, closed_empty_near_object_start=False,
                               protocol=protocol,
                               development_only=True)
        return args, profile

    def test_all_five_audit_entrypoints_dispatch_v9_and_legacy_profiles(self):
        for module_name in self.MODULES:
            for v9 in (True, False):
                with self.subTest(module=module_name, v9=v9), tempfile.TemporaryDirectory() as temp:
                    args, _ = self.fixture(Path(temp), v9)
                    module = importlib.import_module(module_name)
                    dispatched = []

                    def capture(profile):
                        dispatched.append(environment_class_for_profile(profile))
                        raise DispatchReached

                    with patch("argparse.ArgumentParser.parse_args", return_value=args), \
                         patch.object(module, "environment_class_for_profile", side_effect=capture):
                        with self.assertRaises(DispatchReached):
                            module.main()
                    self.assertEqual(dispatched, [BodyGraspEnvironment if v9 else ProfiledManipulationEnv])

    def test_camera_only_derivative_cannot_change_controller_or_feedback_contract(self):
        module = importlib.import_module("VLA.rerender_camera_candidate")
        for key, value in (("gripper_controller", None),
                           ("gripper_state_contract", "commanded_closure"),
                           ("mimic_coupling", "gear"),
                           ("grasp_candidate_scope", "selected_target_legacy")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                args, profile = self.fixture(Path(temp))
                changed = {**profile, key: value}
                args.profile.write_text(json.dumps(changed))
                with patch("argparse.ArgumentParser.parse_args", return_value=args), \
                     patch.object(module, "environment_class_for_profile") as dispatch:
                    with self.assertRaisesRegex(ValueError, "Camera derivative cannot alter " + key):
                        module.main()
                    dispatch.assert_not_called()

    def test_v9_grasp_independence_never_selects_target_filtered_mode(self):
        module = importlib.import_module("VLA.audit_grasp_task_independence")
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            profiles = []

            def capture(profile):
                profiles.append(profile)
                raise DispatchReached

            with patch("argparse.ArgumentParser.parse_args", return_value=args), \
                 patch.object(module, "environment_class_for_profile", side_effect=capture):
                with self.assertRaises(DispatchReached):
                    module.main()
            self.assertEqual([p["grasp_candidate_scope"] for p in profiles], ["all_dynamic_scene_objects"])


if __name__ == "__main__":
    unittest.main()
