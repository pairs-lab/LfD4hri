import unittest
import tempfile
from pathlib import Path
from .replay import attempt_gate
from .prefetch import publish_without_replace
from .events import empty_lift_events
from .ground_context import summarize
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .render import accept_published_cache
from .report import mechanism


class Release(unittest.TestCase):
    def test_incidental_contact_does_not_relabel_success_as_failure(self):
        self.assertTrue(mechanism({'full_put_complete':True,'events':[
            {'mode':'contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce'}]}).startswith('Full PUT'))
    def test_aligned_bowl_attempt_is_not_success(self):
        ok,meta=attempt_gate([.016,0],[0,0],'green bowl',([-.06,-.06,0],[.06,.06,.04]),[.04,.04,.04])
        self.assertTrue(ok);self.assertTrue(meta['actual_support_settle_and_visual_inside_bowl_required'])
    def test_reject_outside_bowl_and_no_arm_correction(self):
        self.assertFalse(attempt_gate([.04,0],[0,0],'green bowl',([-.06,-.06,0],[.06,.06,.04]))[0])
    def test_table_gate_unchanged(self):
        self.assertFalse(attempt_gate([.06,0],[0,0],None)[0])
    def test_atomic_publication_never_replaces_even_empty_primary_directory(self):
        with tempfile.TemporaryDirectory(prefix='r58_atomic_publish_') as tmp:
            root=Path(tmp);source=root/'staging';source.mkdir();target=root/'primary';target.mkdir()
            original_inode=target.stat().st_ino
            self.assertFalse(publish_without_replace(source,target))
            self.assertTrue(source.exists());self.assertEqual(target.stat().st_ino,original_inode)
    def test_atomic_publication_of_completed_new_output(self):
        with tempfile.TemporaryDirectory(prefix='r58_atomic_publish_') as tmp:
            root=Path(tmp);source=root/'staging';source.mkdir();target=root/'new_output'
            inode=source.stat().st_ino;self.assertTrue(publish_without_replace(source,target))
            self.assertFalse(source.exists());self.assertEqual(target.stat().st_ino,inode)

    def test_object_slide_does_not_hide_empty_gripper_lift(self):
        frames=[{'pad_center':[0,0,.03+i*.002],
                 'object_position':[min(.035,i*.001),0,0],
                 'gripper_command':1.,'simulation_time_s':i/30.} for i in range(100)]
        proof={'benchmark_region_success':False,'safety_violation':None}
        events=empty_lift_events(frames,proof,{'held_frames':0})
        self.assertEqual(len(events),1)
        self.assertGreater(events[0]['target_displacement_m'],.025)
        self.assertEqual(events[0]['object_vertical_lift_during_empty_lift_m'],0.)
        self.assertEqual(empty_lift_events(frames,proof,{'held_frames':1}),[])

    def test_carried_object_cannot_be_called_empty_lift(self):
        frames=[{'pad_center':[0,0,.03+i*.002],
                 'object_position':[0,0,i*.002],
                 'gripper_command':1.,'simulation_time_s':i/30.} for i in range(100)]
        self.assertEqual(empty_lift_events(frames,
            {'benchmark_region_success':False,'safety_violation':None},{'held_frames':0}),[])

    def test_high_ground_force_is_reported_without_inventing_a_slam(self):
        with tempfile.TemporaryDirectory(prefix='r58_ground_counterevidence_') as tmp:
            root=Path(tmp)
            atomic_json(root/'physics_event_ticks.json',[
                {'pad_center':[0,0,.2-i*.05/240], 'simulation_time_s':(i+1)/240.,
                 'ground_force_N':400. if 24<=i<36 else 0.,'correct_verified_hold':False}
                for i in range(72)])
            report=summarize(root)
            self.assertEqual(report['peak_gripper_ground_force_N'],400.)
            self.assertAlmostEqual(report['events'][0]['entry_vertical_velocity_50ms_m_per_s'],-.05)
            self.assertTrue(report['force_alone_does_not_establish_high_speed_impact_or_root_cause'])
            self.assertEqual(report['verified_correct_hold_ticks_240Hz'],0)

    def test_cache_publication_race_accepts_only_matching_closed_output(self):
        with tempfile.TemporaryDirectory(prefix='r58_cache_race_') as tmp:
            root=Path(tmp);cap=root/'capture';cap.mkdir();out=root/'published';out.mkdir()
            atomic_json(cap/'recorded_physics_states.json',[{'physics_tick':8}])
            atomic_json(cap/'initial_scene.json',{'seed':7})
            atomic_json(out/'initial_scene.json',{'seed':7})
            self.assertFalse(accept_published_cache(out,cap))
            atomic_json(out/'render_complete.json',{'all_writers_closed':True,'all_golden_RGB_and_seg_checks_passed':True})
            atomic_json(out/'renderer.json',{'all_per_rollout_golden_exact':True,
                'source_state_sha256':sha256_file(cap/'recorded_physics_states.json')})
            self.assertTrue(accept_published_cache(out,cap))
            atomic_json(out/'initial_scene.json',{'seed':8})
            with self.assertRaises(ValueError):accept_published_cache(out,cap)


if __name__=='__main__':unittest.main()
