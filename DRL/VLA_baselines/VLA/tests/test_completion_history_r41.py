import json
from pathlib import Path
import tempfile
import unittest
from VLA.audit_smol_task_completion_r41 import analyze


class CompletionHistoryTests(unittest.TestCase):
    def make(self,root):
        state=[0.]*64;state[18]=state[22]=1.;state[24:27]=[.04]*3
        trajectory=root/'trajectory.json'
        frame={'pre_action_scene64':state,'success':True,'before':{'correct_object_held':False},
            'after':{'correct_object_held':True}}
        trajectory.write_text(json.dumps({'move':[frame]}))
        row={'seed':1,'task':'move','success':True,'steps':1,'safety_violation':None,'trajectory':str(trajectory)}
        progress=root/'progress.json'
        progress.write_text(json.dumps({'completed_trials':1,'planned_trials':1,'rows':[row]}))
        return progress

    def test_retained_attachment_not_mislabelled_as_bilateral(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=self.make(root);analyze(file,root/'audit')
            result=json.loads((root/'audit/audit.json').read_text())
            self.assertTrue(result['cases'][0]['correct_attachment_observed'])
            self.assertFalse(result['cases'][0]['verified_correct_bilateral_BEFORE_action_observed'])
            self.assertEqual(result['by_task']['move']['predeclared_successes'],1)
            self.assertTrue(result['no_primary_scores_actions_horizons_or_physics_changed'])

    def test_partial_or_mismatched_results_never_certified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=self.make(root);data=json.loads(file.read_text());data['planned_trials']=2
            file.write_text(json.dumps(data))
            with self.assertRaises(ValueError):analyze(file,root/'audit')


if __name__=='__main__':unittest.main()
