import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from VLA.collect_onpolicy_handoff_pilot import StudentPrefixReset
from VLA.common.tasks import TaskSpec


class StudentPrefixTests(unittest.TestCase):
    def fixture(self, *, aborted=False):
        task = TaskSpec(0,"touch","Touch the red block","red block",(0.,-.5,.03),max_steps=80)
        row = {"task": json.loads(json.dumps(task.to_dict()))}
        obs = SimpleNamespace(step=0)
        env = SimpleNamespace(requested_arm_targets=np.zeros(6),step_count=20,success_streak=0,_last_success=False)
        env.step = lambda action: (obs,0.,aborted,{"executed_canonical_action":action.tolist(),"safety_violation":"guard" if aborted else None})
        calls = []
        policy = SimpleNamespace(reset=lambda request:calls.append(request), predict=lambda sensor:np.array([0,0,-.02,0,0,0,-1],np.float32))
        return task, StudentPrefixReset(env,policy,"/train",row,20), obs, calls

    @patch("VLA.collect_onpolicy_handoff_pilot._interaction_audit", return_value={"held_object":None})
    def test_json_list_vs_dataclass_tuple_is_not_false_instruction_mismatch(self,audit):
        task, wrapper, obs, calls = self.fixture()
        with patch("VLA.collect_onpolicy_handoff_pilot.reset_source_episode",return_value=obs):
            self.assertIs(wrapper.reset(task),obs)
        self.assertEqual(calls,[task])
        self.assertEqual(len(wrapper.actions),20)
        self.assertEqual(len(wrapper.native_actions),20)
        self.assertEqual(wrapper.held_before,[])
        self.assertEqual(wrapper.env.step_count,0)

    @patch("VLA.collect_onpolicy_handoff_pilot._interaction_audit", return_value={"held_object":None})
    def test_aborted_prefix_is_retained_not_retried_or_replaced(self,audit):
        task, wrapper, obs, _ = self.fixture(aborted=True)
        with patch("VLA.collect_onpolicy_handoff_pilot.reset_source_episode",return_value=obs):
            with self.assertRaisesRegex(ValueError,"terminated/aborted"):
                wrapper.reset(task)
        self.assertEqual(len(wrapper.actions),1)

    @patch("VLA.collect_onpolicy_handoff_pilot._interaction_audit", return_value={"held_object":"wrong block"})
    def test_unsupported_held_prefix_cannot_be_called_empty(self,audit):
        task, wrapper, obs, _ = self.fixture()
        with patch("VLA.collect_onpolicy_handoff_pilot.reset_source_episode",return_value=obs):
            with self.assertRaisesRegex(ValueError,"NOT-held"):
                wrapper.reset(task)


if __name__ == "__main__":
    unittest.main()
