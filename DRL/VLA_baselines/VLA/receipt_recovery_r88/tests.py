import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from VLA.dita800_r87.evaluate import authorized_frozen
from .evaluate import compile_resume,ledger,SOURCE,OUT


class Contracts(unittest.TestCase):
    def test_completed_ledger_unique_and_partial_excluded(self):
        prior,rows,completed=ledger(SOURCE)
        self.assertEqual(len(rows),243);self.assertEqual(len(completed),243)
        self.assertEqual(prior['motor_controls']-sum(r['steps'] for r in rows),174)
        self.assertNotIn((20284061,'pick_place'),completed)
    def test_exact_cross_version_reference(self):
        sim=json.loads((OUT/'SIM_MAP_REFERENCE.json').read_text())
        dita=json.loads((OUT/'DITA_MAP_REFERENCE.json').read_text())
        self.assertEqual(sim['cases'],271);self.assertEqual(sim['cases'],dita['cases'])
        self.assertEqual(sim['reference_images_combined_sha256'],dita['reference_images_combined_sha256'])
        self.assertEqual(dita['reference_vs_original_differing_cases'],0)
    def test_failed_request_matches_reproduced_native_map(self):
        replay=json.loads((OUT/'REPLAY_PROOF.json').read_text())
        pair=json.loads((OUT/'CROSS_VERSION_MAP.json').read_text())
        self.assertTrue(replay['passed']);self.assertEqual(replay['matched_actual_request_hashes'],381)
        self.assertEqual(pair['differing_pixels'],4)
        sim=json.loads((OUT/'SIM_MAP_REFERENCE.json').read_text())
        self.assertEqual(sim['failing_reference_sha256'],pair['Dita_map_sha256'])
    def test_only_resume_eligibility_and_reference_injection(self):
        from VLA.autonomy_r83 import evaluate as old
        fn=compile_resume(old,lambda:([],0,0),lambda c:[],lambda *a:False,SimpleNamespace(Session=None),lambda *a:None)
        self.assertTrue(callable(fn))
        self.assertIs(fn.__globals__['environment_class_for_profile'],old.environment_class_for_profile)
        self.assertIs(fn.__globals__['model_input'],old.model_input)
    def test_user_authorization_does_not_claim_confirmation_passed(self):
        method={'frozen_after_confirmation':False,'frozen_for_user_requested_benchmark':True,
            'benchmark_authorization':{'user_requested_800_despite_failed_pilot_gate':True}}
        self.assertTrue(authorized_frozen(method,{'role':'fresh_secondary_final800','seeds':list(range(20284001,20284201))}))


if __name__=='__main__':unittest.main()
