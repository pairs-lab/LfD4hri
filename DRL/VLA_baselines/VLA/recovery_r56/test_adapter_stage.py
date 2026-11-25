"""CPU schedule proof; real GPU2/save/reload still required before long fit."""
import math
import unittest
from .adapter_fit import stage_scale,native_cosine


class Schedule(unittest.TestCase):
    def test_bounded_warmup_peak_and_floor(self):
        self.assertAlmostEqual(stage_scale(0),.109)
        self.assertAlmostEqual(stage_scale(99),1.)
        self.assertAlmostEqual(stage_scale(100),1.)
        self.assertAlmostEqual(stage_scale(1500),.1)
        self.assertTrue(all(.1<=stage_scale(i)<=1. for i in range(1501)))

    def test_cumulative_counter_is_rebased_only_for_lr(self):
        origin=9000;total=10500
        for actual in range(origin,total+1):
            global_angle=math.pi*(actual-200)/(total-200)
            native_scale=.1+.9*.5*(1+native_cosine(global_angle,origin))
            self.assertAlmostEqual(native_scale,stage_scale(actual-origin),places=10)

    def test_continuing_gate_does_not_restart_twice(self):
        origin=9000;gate=9002;total=10500
        global_angle=math.pi*(gate-200)/(total-200)
        scale=.1+.9*.5*(1+native_cosine(global_angle,origin))
        self.assertAlmostEqual(scale,stage_scale(2))
        self.assertNotAlmostEqual(scale,stage_scale(0))

    def test_out_of_budget_rejected(self):
        for position in (-1,1501):
            with self.assertRaises(ValueError):stage_scale(position)


if __name__=='__main__':unittest.main()
