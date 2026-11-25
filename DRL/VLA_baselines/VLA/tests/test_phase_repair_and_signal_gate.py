import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from VLA.smolvla.phase_balanced_sampling import label_phases, mix_phase_weights
from VLA.smolvla.audit_target_grounding import ordinary_target_pairs_from_mixed_train
from VLA.rdt.signal_qualification import require_signal_qualification


class PhaseRecipeTests(unittest.TestCase):
    def data(self):
        actions = np.zeros((6,7), np.float32)
        actions[:,6] = -1
        actions[0,2] = -.02
        actions[1,0] = .03
        actions[2,6] = 1
        actions[3,2] = .01
        actions[5,6] = 1
        return {"canonical_actions": actions, "gripper_closure": np.array([0.,0.,0.,.5,.99,0.]),
            "next_grasped_object": np.array([False,False,True,False,False,False]),
            "next_verified_grasp": np.array([False,False,True,False,False,False])}

    def test_actual_feedback_not_close_command_defines_acquisition(self):
        phases = label_phases(self.data())
        self.assertEqual(phases.tolist(), ["descent","xy_choice","acquisition","held_transport","closed_unheld_open","other"])

    def test_every_frame_retained_with_equal_task_and_target_mass(self):
        rows, labels = [], []
        for task in ("touch","pick","move","pick_place"):
            for target in ("red", "blue"):
                rows.append({"task":{"name":task,"pick_name":target}, "steps":6})
                labels.append(label_phases(self.data()))
        weights = mix_phase_weights(rows, labels)
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(weights.sum(), 1.)
        for start in range(0,len(weights),6):
            self.assertAlmostEqual(weights[start:start+6].sum(), .125)
        self.assertGreater(weights[1], weights[0])

    def test_unknown_or_misaligned_phases_rejected(self):
        rows = [{"task":{"name":task,"pick_name":"red"},"steps":2} for task in ("touch","pick","move","pick_place")]
        with self.assertRaises(ValueError): mix_phase_weights(rows, [np.array(["unknown","other"])]*4)
        with self.assertRaises(ValueError): mix_phase_weights(rows, [])

    def test_nonfinite_feedback_is_not_sampling_evidence(self):
        data = self.data(); data["gripper_closure"][0] = np.nan
        with self.assertRaises(ValueError): label_phases(data)

    def test_mixed_grounding_selects_all_declared_ordinary_pairs_not_recoveries(self):
        rows = []
        for task in ("touch","pick","move","pick_place"):
            for variant in (0,1):
                rows.append({"grounding_text_intervention":"target", "variant":variant,
                    "canonical_source":"train", "paired_scene_source":task,
                    "task":{"name":task,"pick_name":str(variant),"prompt":str(variant)}})
        rows.append({"task":{"name":"pick"},"grounding_text_intervention":"target", "recovery_start_actions_not_training_labels":[[0]*7]})
        groups = ordinary_target_pairs_from_mixed_train(rows)
        self.assertEqual(len(groups),4)
        with self.assertRaises(ValueError): ordinary_target_pairs_from_mixed_train(rows[:-2])


class SignalQualificationTests(unittest.TestCase):
    def test_old_bf16_default_and_two_update_smoke_do_not_require_new_proof(self):
        self.assertIsNone(require_signal_qualification(SimpleNamespace(signal_precision="bf16",steps=5000)))
        self.assertIsNone(require_signal_qualification(SimpleNamespace(signal_precision="fp32_state_action_readout_latent_v1",steps=2)))

    def test_long_fp32_fit_cannot_bypass_missing_proof(self):
        with self.assertRaisesRegex(ValueError,"explicit successful"):
            require_signal_qualification(SimpleNamespace(signal_precision="fp32_state_action_readout_latent_v1",steps=5000))

    def test_failed_or_success_claiming_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"proof.json"
            args = SimpleNamespace(signal_precision="fp32_state_action_readout_latent_v1",steps=5000,signal_qualification=path)
            for data in ({"passed":False}, {"passed":True,"scope":"robot_success"}):
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,"Wrong/failed"):
                    require_signal_qualification(args)


if __name__ == "__main__":
    unittest.main()
