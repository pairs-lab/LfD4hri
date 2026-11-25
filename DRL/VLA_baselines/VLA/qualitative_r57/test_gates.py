import unittest
import json
import numpy as np
from .release import landing_gate,complete
from .events import classify
from .selection import assess_vla


def sample(i,held=False,x=0.,z=.03,speed=0.,force=0.):
    return {'simulation_time_s':i/30,'object_position':[x,0,z],'object_linear_velocity':[0,0,speed],
        'correct_verified_hold':held,'goal_position':[0,0,.03],
        'destination_support_force_N':force,'gripper_command':1. if held else -1.}


class Gates(unittest.TestCase):
    def test_region_does_not_accept_held_put(self):
        frames=[sample(i,held=True,force=1.) for i in range(40)]
        self.assertFalse(classify(frames,[])['full_put_complete'])
    def test_support_settle_and_real_prior_hold_required(self):
        frames=[sample(i,held=i<6,force=1. if i>=6 else 0.) for i in range(40)]
        self.assertTrue(classify(frames,[])['full_put_complete'])
        frames=[sample(i,force=1.) for i in range(40)]
        self.assertFalse(classify(frames,[])['full_put_complete'])
    def test_fall_without_hold_is_not_in_transit_drop(self):
        frames=[sample(i,z=.1-i*.002,speed=-.2) for i in range(30)]
        self.assertFalse(any(e['mode']=='verified_in_transit_drop' for e in classify(frames,[])['events']))
    def test_release_at_goal_is_not_drop(self):
        frames=[sample(i,held=i<6,z=.1 if i<6 else .03,force=1. if i>=6 else 0.) for i in range(30)]
        self.assertFalse(any(e['mode']=='verified_in_transit_drop' for e in classify(frames,[])['events']))
    def test_drop_away_from_goal(self):
        frames=[sample(i,held=i<10,x=.2,z=.03 if i<3 else .15 if i<10 else .03) for i in range(30)]
        events=classify(frames,[])['events']
        self.assertTrue(any(e['mode']=='verified_in_transit_drop' for e in events))
    def test_safety_without_contact_is_not_slam(self):
        result=classify([sample(i) for i in range(30)],[])
        self.assertFalse(any('slam' in e['mode'] for e in result['events']))
    def test_no_cognitive_failure_claim(self):
        self.assertFalse(classify([sample(i) for i in range(30)],[])['cognitive_misidentification_or_memory_loss_proven'])
    def test_bilateral_sensor_loss_with_attachment_is_not_drop(self):
        frames=[sample(i,held=i<10,x=.2,z=.03 if i<3 else .15 if i<10 else .1) for i in range(30)]
        for row in frames:row['correct_attached']=True
        self.assertFalse(any(e['mode']=='verified_in_transit_drop' for e in classify(frames,[])['events']))
    def test_contact_loss_with_attachment_is_not_completed_put(self):
        frames=[sample(i,held=i<6,force=1.) for i in range(40)]
        for row in frames:row['correct_attached']=True
        self.assertFalse(classify(frames,[])['full_put_complete'])
    def test_drop_event_is_json_serializable(self):
        frames=[sample(i,held=i<10,x=.2,z=.03 if i<3 else .15 if i<10 else .03) for i in range(30)]
        json.dumps(classify(frames,[]),allow_nan=False)
    def test_release_gate_rejects_misaligned_table(self):
        self.assertFalse(landing_gate([.06,0],[0,0],None)[0])
    def test_release_gate_does_not_treat_outer_bowl_box_as_opening(self):
        self.assertFalse(landing_gate([.08,0],[0,0],'red bowl',([-.1,-.1,0],[.1,.1,.1]),[.06,.06,.06])[0])
    def test_terminal_metrics_require_actual_support(self):
        values=[{'correct_attached':False,'goal_XY_m':0.,'support_force_N':0.,
            'speed_m_per_s':0.,'gripper_command':-1.} for _ in range(5)]
        self.assertFalse(complete(values,True))
    def test_release_only_failure_is_not_a_strong_clip(self):
        evidence={'full_put_complete':False,'events':[]}
        self.assertFalse(assess_vla(evidence,{'benchmark_region_success':True})['eligible_for_render'])
    def test_camera_loss_and_guard_stall_are_not_sufficient(self):
        evidence={'full_put_complete':False,'events':[
            {'mode':'policy_wrist_target_absent_and_measured_pose_stall_NOT_memory_loss'}]}
        self.assertFalse(assess_vla(evidence,{'benchmark_region_success':False})['eligible_for_render'])
    def test_strong_contact_failure_is_only_a_candidate(self):
        evidence={'full_put_complete':False,'events':[{
            'mode':'contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce',
            'force_N':12.7,'object_displacement_within250ms_m':.265,
            'maximum_object_speed_m_per_s':1.72}]}
        result=assess_vla(evidence,{'benchmark_region_success':False})
        self.assertTrue(result['eligible_for_render'])
        self.assertTrue(result['manual_encoded_visual_review_required'])
        self.assertFalse(assess_vla(evidence,{'benchmark_region_success':False,
            'safety_violation':'unsafe linkage'})['eligible_for_render'])
    def test_small_first_contact_does_not_hide_later_strong_impact(self):
        dense=[]
        for i in range(300):
            x=.04*np.clip((i-20)/24,0,1)+.2*np.clip((i-150)/60,0,1)
            speed=.4 if 20<=i<44 else .8 if 150<=i<210 else 0.
            dense.append({'object_velocity':[speed,0,0],'object_position':[float(x),0,.03],
                'target_force_N':2. if i in (20,150) else 0.,'correct_verified_hold':False,
                'simulation_time_s':i/240})
        result=classify([sample(i) for i in range(45)],dense)
        self.assertEqual(len(result['events']),2)
        self.assertTrue(assess_vla(result,{'benchmark_region_success':False})['eligible_for_render'])


if __name__=='__main__':unittest.main()
