import copy
import unittest
from VLA.collect_recorded_miss_corrections import eligible, verify_recreated
from VLA.common.tasks import TaskSpec, select_tasks
import json


class RecordedMissCorrectionTest(unittest.TestCase):
    def test_json_roundtrip_keeps_task_identity_despite_list_tuple_storage(self):
        task = select_tasks("pick")[0]
        self.assertEqual(task, TaskSpec.from_dict(json.loads(json.dumps(task.to_dict()))))

    def state(self):
        return {"closed_empty_proxy": True, "surface_clearance_m": .009, "gripper_contacts": [],
                "held_body_id": None, "joint_positions_rad": [0]*6, "mount_position_m": [0]*3,
                "tcp_position_m": [0]*3, "target_position_m": [.1]*3,
                "measured_encoder_fk_closure": .97, "jaw_gap_m": .005}

    def test_eligibility_requires_safe_real_feedback_not_close_intent(self):
        frame = {"after": self.state(), "safety_abort": False}
        self.assertTrue(eligible(frame))
        for key, value in (("closed_empty_proxy", False), ("surface_clearance_m", .007), ("gripper_contacts", [{}])):
            changed = copy.deepcopy(frame); changed["after"][key] = value
            self.assertFalse(eligible(changed))

    def test_replay_state_must_match_exactly_and_be_unheld(self):
        actual = self.state(); verify_recreated(actual, self.state())
        actual["joint_positions_rad"][0] = .01
        with self.assertRaises(ValueError):
            verify_recreated(actual, self.state())
        actual = self.state(); actual["held_body_id"] = 8
        with self.assertRaises(ValueError):
            verify_recreated(actual, self.state())
