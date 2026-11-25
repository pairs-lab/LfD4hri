import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
from .extend_pick import retime,ExtendedPick,FINAL_HOLD_SECONDS
from .motion import RequestedRollout


class ExtendedPickTests(unittest.TestCase):
    def test_slow_physical_clock_preserves_entire_geometry(self):
        curve=SimpleNamespace(duration=3.,sample=lambda t:np.asarray([t,t*t]))
        new=retime(curve,4.5)
        for fraction in (0.,.25,.5,1.):
            np.testing.assert_allclose(new.sample(fraction*4.5),curve.sample(fraction*3.))
        with self.assertRaises(ValueError):retime(curve,2.)

    def test_only_final_hold_is_extended(self):
        obj=ExtendedPick.__new__(ExtendedPick)
        with mock.patch.object(RequestedRollout,'hold') as original:
            obj.hold(None,None,1.2,'final',1.)
            self.assertEqual(original.call_args.args[2],FINAL_HOLD_SECONDS)
            obj.hold(None,None,.8,'grasp',1.)
            self.assertEqual(original.call_args.args[2],.8)


if __name__=='__main__':unittest.main()
