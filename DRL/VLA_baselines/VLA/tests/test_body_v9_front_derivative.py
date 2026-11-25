import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from VLA.common.observation_profile import profile_hash
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.rerender_body_v9_front_candidate import front_only_profile, main, numeric_check, successful_raw_rows, wrist_check


class FrontDerivativeTest(unittest.TestCase):
    def profile(self):
        return json.loads((Path(__file__).resolve().parents[1]/"profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())

    def test_only_front_camera_and_name_can_change(self):
        old = self.profile()
        new = copy.deepcopy(old); new["name"] += "_newfront"
        new["cameras"]["front"]["eye_world"][0] += .1
        self.assertEqual(front_only_profile(old,new),new)
        for key,value in (("home_joints",[0.]*6),("gripper_controller","legacy"),("extra_physics_option",True)):
            changed = copy.deepcopy(new); changed[key] = value
            with self.assertRaises(ValueError):
                front_only_profile(old,changed)
        for camera in ("wrist","overview"):
            changed = copy.deepcopy(new); changed["cameras"][camera]["fov_degrees"] += 1
            with self.assertRaises(ValueError):
                front_only_profile(old,changed)

    def test_failed_or_paired_source_is_rejected(self):
        identity = profile_hash(self.profile())
        row = {"success":True,"observation_profile_sha256":identity,
               "success_contract":{"required_consecutive_steps":10}}
        successful_raw_rows([row],identity,10)
        for change in ({"success":False},{"safety_violation":"abort"},{"paired_scene_source":"x"},
                       {"observation_profile_sha256":"different"}):
            with self.assertRaises(ValueError):
                successful_raw_rows([{**row,**change}],identity,10)

    def maxima(self):
        return {"numeric_max_abs_error":0.,"wrist_max_channel_lsb":0,"wrist_max_changed_pixel_fraction":0.}

    def test_paired_reset_is_explicit_and_validated(self):
        identity = profile_hash(self.profile())
        row = {"success": True, "observation_profile_sha256": identity,
               "success_contract": {"required_consecutive_steps": 1},
               "paired_scene_source": "original.npz", "variant": 0,
               "paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget",
               "recovery_start_actions_not_training_labels": [[0.] * 7]}
        with self.assertRaises(ValueError):
            successful_raw_rows([row], identity, 1)
        successful_raw_rows([row], identity, 1, True)
        for extra in ({"variant": 2}, {"paired_reset_strategy": "unknown"}, {"paired_scene_source": ""}):
            with self.assertRaises(ValueError):
                successful_raw_rows([{**row, **extra}], identity, 1, True)

    def test_numeric_nonfinite_shapes_and_feedback_mismatch_fail_closed(self):
        numeric_check([.1,.2],[.1,.2],"state",self.maxima())
        for actual,expected in (([np.nan],[0.]),([0.],[np.nan]),([0.,1.],[0.]),([.00001],[0.])):
            with self.assertRaises(RuntimeError):
                numeric_check(actual,expected,"state",self.maxima())

    def test_all_wrist_masks_exact_and_only_small_lsb_rounding_allowed(self):
        old = np.zeros((100,100,3),dtype=np.uint8)
        masks = {kind:np.zeros((100,100),dtype=bool) for kind in ("pick","place","robot")}
        rounded = old.copy(); rounded[0,0,0] = 1
        wrist_check(rounded,old,masks,masks,"wrist",self.maxima())
        for kind in masks:
            changed = {key:value.copy() for key,value in masks.items()}; changed[kind][0,0] = True
            with self.assertRaises(RuntimeError):
                wrist_check(old,old,changed,masks,"wrist",self.maxima())
        too_much = old.copy(); too_much[:2,:,0] = 1
        with self.assertRaises(RuntimeError):
            wrist_check(too_much,old,masks,masks,"wrist",self.maxima())
        two_lsb = old.copy(); two_lsb[0,0,0] = 2
        with self.assertRaises(RuntimeError):
            wrist_check(two_lsb,old,masks,masks,"wrist",self.maxima())

    def test_interrupted_derivative_resumes_verified_prefix_without_rewriting(self):
        # Real indexed payload hashes/checkpoints, mocked physics replay only.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/"source"; source.mkdir()
            output = root/"output"; old = self.profile(); new = copy.deepcopy(old)
            new["name"] += "_front_derivative_test"
            new["cameras"]["front"]["eye_world"][0] += .1
            candidate = root/"profile.json"; candidate.write_text(json.dumps(new))
            rows = [{"file":f"ep{i}.npz","success":True,"steps":1,"task":{"name":"touch"},
                     "observation_profile_sha256":profile_hash(old),
                     "success_contract":{"required_consecutive_steps":10}} for i in range(2)]
            (source/"index.json").write_text(json.dumps(rows))
            manifest = {"robot":"ur5e","image_size":384,"seed":42,"required_success_steps":10,
                        "observation_profile":old,"observation_profile_sha256":profile_hash(old),
                        "environment_runtime_fingerprint":environment_runtime_fingerprint(old)}
            (source/"dataset_manifest.json").write_text(json.dumps(manifest))
            for i in range(2):
                np.savez_compressed(source/f"ep{i}.npz",value=np.array([i]))
            arguments = ["rerender","--source",str(source),"--profile",str(candidate),"--output",str(output)]
            def interrupt(env,item,arrays,maxima,reset_source=None):
                if item["file"] == "ep1.npz":
                    raise RuntimeError("simulated interruption")
                return arrays
            with patch("sys.argv",arguments), patch("VLA.rerender_body_v9_front_candidate.BodyGraspEnvironment"), \
                 patch("VLA.rerender_body_v9_front_candidate.replay_episode",side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                    main()
            self.assertTrue((output/"ep0.npz").is_file())
            self.assertFalse((output/"ep1.npz").exists())
            committed_bytes = (output/"ep0.npz").read_bytes()
            state = json.loads((output/"collection_state.json").read_text())
            self.assertEqual(state["status"],"failed_partial_preserved")
            with patch("sys.argv",arguments+["--resume"]), patch("VLA.rerender_body_v9_front_candidate.BodyGraspEnvironment"), \
                 patch("VLA.rerender_body_v9_front_candidate.replay_episode",side_effect=lambda env,item,arrays,maxima,reset_source=None:arrays) as replay:
                main()
            self.assertEqual(replay.call_count,1)
            self.assertEqual((output/"ep0.npz").read_bytes(),committed_bytes)
            self.assertEqual(len(json.loads((output/"index.json").read_text())),2)
            derivative = json.loads((output/"dataset_manifest.json").read_text())
            self.assertTrue(derivative["candidate_only"])
            self.assertFalse(derivative["training_certified"])
            damaged = json.loads((output/"index.json").read_text()); damaged[0]["task"]["name"] = "pick"
            (output/"index.json").write_text(json.dumps(damaged))
            with patch("sys.argv",arguments+["--resume"]):
                with self.assertRaisesRegex(RuntimeError,"metadata/labels"):
                    main()


if __name__ == "__main__":
    unittest.main()
