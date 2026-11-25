from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import ast
import inspect
import numpy as np
from VLA.openvla.recovery_balanced import sampling_weights, stratum,cap_stationary


class MotionBalanceTests(unittest.TestCase):
    def test_trainer_explicitly_selects_full64_not_historical18_schema(self):
        from VLA.openvla.recovery_balanced import train
        tree=ast.parse(inspect.getsource(train))
        assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)]
        self.assertTrue(any(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
            and target.value.id=='privileged' and target.attr=='PROMPT_SCHEMA'
            and isinstance(node.value,ast.Name) and node.value.id=='PROMPT_SCHEMA'
            for node in assignments for target in node.targets))

    def test_strata_use_only_offline_action_and_current_hold(self):
        state = np.zeros(64, np.float32)
        action = np.asarray([.03,0,0,0,0,0,-1],np.float32)
        self.assertEqual(stratum(action,state),'open_xy_or_yaw_alignment')
        action[6] = 1
        self.assertEqual(stratum(action,state),'close_without_current_verified_hold')
        state[17] = 1
        self.assertEqual(stratum(action,state),'held_transport')

    def test_all_four_tasks_and_rare_motion_strata_have_exact_mass(self):
        class Original:
            def __len__(self): return 12
        with tempfile.TemporaryDirectory() as directory:
            original=Original();original.source=Path(directory)
            original.rows=[];original.states=np.zeros((12,64),np.float32);original.ends=[3,6,9,12]
            correction=SimpleNamespace(rows=[])
            for i,task in enumerate(('touch','pick','move','pick_place')):
                path=original.source/f'{i}.npz'
                np.savez(path,canonical_actions=np.asarray([[.03,0,0,0,0,0,-1],[.03,0,0,0,0,0,-1],[0,0,-.02,0,0,0,-1]],np.float32))
                original.rows.append({'file':path.name,'steps':3,'task':{'name':task,'pick_name':'object'}})
                correction.rows.append({'steps':2,'task':{'name':task}})
            weights,proof=sampling_weights(original,correction)
            self.assertAlmostEqual(weights[:12].sum(),.8)
            self.assertAlmostEqual(weights[12:].sum(),.2)
            self.assertAlmostEqual(weights[0]+weights[1],weights[2])
            self.assertAlmostEqual(weights[:3].sum(),.2)
            self.assertFalse(proof['runtime_teacher_phase_injected'])

    def test_rare_wait_does_not_get_one_third_of_training_probability(self):
        weights=np.array([1/3,1/3,1/3]);labels=[('touch','object',s) for s in ('open_wait','open_xy_or_yaw_alignment','open_vertical_reach')]
        adjusted,proof=cap_stationary(weights,labels)
        self.assertAlmostEqual(adjusted[0],.05);self.assertAlmostEqual(adjusted[1],.475)
        self.assertAlmostEqual(adjusted.sum(),1);self.assertTrue(proof['meaningful_reopen_release_NOT_capped'])

    def test_physical_reopening_and_release_are_not_misclassified_as_noop_waits(self):
        state=np.zeros(64,np.float32);state[6]=.99;state[10]=1
        action=np.array([0,0,0,0,0,0,-1],np.float32)
        self.assertEqual(stratum(action,state),'near_target_reopen')
        state[13]=-.5;self.assertEqual(stratum(action,state),'reopen_from_hover')
        state[17]=1;self.assertEqual(stratum(action,state),'release_current_hold')
        state[17]=0;state[30]=1;self.assertEqual(stratum(action,state),'release_wrong_held_body')


if __name__ == '__main__':
    unittest.main()
