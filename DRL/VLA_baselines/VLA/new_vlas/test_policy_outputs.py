import ast
from pathlib import Path
import unittest

import numpy as np

from VLA.common.robotiq import Robotiq2F85
from VLA.new_vlas.policy_outputs import native_gripper_command


class NativeGripperTests(unittest.TestCase):
    def test_adapter_matches_actual_official_postprocessing(self):
        path = Path(__file__).resolve().parents[1]/'third_party/vla_adapter/experiments/robot/robot_utils.py'
        selected = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
            and node.name in ('normalize_gripper_action', 'invert_gripper_action')]
        namespace = {'np': np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        for value in (0., .2, .45, .5, .55, .8, 1.):
            action = np.array([0, 0, 0, 0, 0, 0, value], np.float32)
            actual = namespace['invert_gripper_action'](namespace['normalize_gripper_action'](action, binarize=True))
            self.assertEqual(native_gripper_command(value, convention='adapter_open01'), actual[-1])

    def test_native_binarization_does_not_get_stuck_in_existing_motor_deadband(self):
        gripper = Robotiq2F85.__new__(Robotiq2F85)
        gripper.command = -1.
        gripper.set_command(1-2*.45)
        self.assertEqual(gripper.command, -1.)
        gripper.set_command(native_gripper_command(.45, convention='adapter_open01'))
        self.assertEqual(gripper.command, 1.)
        self.assertEqual(native_gripper_command(.55, convention='xvla_closed01'), 1.)


if __name__ == '__main__':
    unittest.main()
