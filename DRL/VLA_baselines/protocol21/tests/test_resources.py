import tempfile
import unittest
from protocol21.resources import check
from protocol21.native import create
from protocol21.specs import SPECS
from protocol21.tests.test_inputs import stats


class ResourceTests(unittest.TestCase):
    def test_missing_foundations_never_fall_back_to_a_fake_policy(self):
        with tempfile.TemporaryDirectory() as root:
            for family in SPECS:
                with self.subTest(family=family):
                    r=check(family,root)
                    self.assertFalse(r['local_files_present'])
                    self.assertFalse(r['native_forward_verified'])
                    with self.assertRaises(FileNotFoundError):create(family,stats(),resources=root)
