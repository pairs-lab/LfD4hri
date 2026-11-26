import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from protocol21.inputs import Frame, Inputs, prefix
from protocol21.specs import SPECS
from protocol21.state21_reference import pack, fit_train_stats, normalize
from protocol21.data import Bank,digest


def frame(time=0., **kw):
    raw = pack(q=np.arange(6)*.1,closure01=.3,pad_midpoint_world=[.1,-.5,.2],
               previous_signed_close=-1,tool_to_world_rotation=np.eye(3),surface_jaw_gap_m=.04,
               left_external_normal_force_N=.3,right_external_normal_force_N=.05)
    image = np.full((32,32,3),130,np.uint8)
    values = dict(raw_state21=raw,ee_position=np.array([.1,-.5,.35],np.float32),front=image,
                  wrist=image.copy(),instruction='put red block into blue bowl',time_s=time,
                  depth_z_m=np.ones((32,32),np.float32),intrinsics=np.eye(3,dtype=np.float32),
                  extrinsics_OpenCV=np.eye(4,dtype=np.float32))
    values.update(kw)
    return Frame(**values)


def stats():
    return {'robot21':fit_train_stats([frame().raw_state21]),
            'adapter':{'proprio':{'q01':[-1]*8,'q99':[1]*8,'mask':[True]*8}}}


class InputsTests(unittest.TestCase):
    def test_all_ten_actual_widths(self):
        for family,spec in SPECS.items():
            with self.subTest(family=family):
                p=Inputs(family,stats());p.push(frame());s=p.sample()
                if family=='seer':
                    self.assertEqual(s['state']['native7'].shape,(7,))
                    self.assertEqual(s['robot21_history'].shape,(7,21))
                else:self.assertEqual(s['state'].shape,(spec.state_width,))
                self.assertNotIn('map',s);self.assertNotIn('raw64',s);self.assertNotIn('numeric67',s)

    def test_bits_survive_normalization(self):
        f=frame();p=Inputs('flower',stats());p.push(f)
        np.testing.assert_array_equal(p.sample()['state'][18:20],f.raw_state21[18:20])

    def test_native_prefixes(self):
        x=frame();x20=prefix(x,'xvla')
        np.testing.assert_array_equal(x20[10:],np.zeros(10))
        np.testing.assert_allclose(x20[:3],x.ee_position)
        np.testing.assert_allclose(prefix(x,'vla_adapter')[-2:],[.02,-.02])
        self.assertEqual(prefix(x,'seer')[-1],1)

    def test_native_prefix_not_pad_midpoint(self):
        x=frame();self.assertNotEqual(x.ee_position[2],x.raw_state21[9])
        np.testing.assert_array_equal(prefix(x,'xvla')[:3],x.ee_position)

    def test_reset_padding_and_causal_history(self):
        p=Inputs('dita',stats());p.push(frame());p.push(frame(.1))
        self.assertEqual(p.sample()['history_times'],[0.,.1])
        p.reset();p.push(frame());self.assertEqual(p.sample()['history_times'],[0.,0.])

    def test_skipped_control_history_rejected(self):
        p=Inputs('seer',stats());p.push(frame())
        with self.assertRaises(ValueError):p.push(frame(.3))

    def test_old_width_rejected(self):
        with self.assertRaises(ValueError):frame(raw_state21=np.zeros(67)).checked()
        s=stats();s['robot21']['mean']=[0]*67
        with self.assertRaises(ValueError):Inputs('flower',s)

    def test_binary_normalizer_rejected(self):
        s=stats();s['robot21']['std'][18]=2
        with self.assertRaises(ValueError):normalize(frame().raw_state21,s['robot21'])

    def test_pi0_text_and_zero_padding(self):
        from protocol21.pi0 import payload
        p=Inputs('pi0',stats());p.push(frame());s=p.sample();v=payload(s)
        self.assertEqual(len(v['image']),2)
        np.testing.assert_array_equal(v['state'][21:],np.zeros(11))
        self.assertIn('total_pad_normal_force_N=0.3500',s['prompt'])
        self.assertNotIn('target_size',s['prompt'])

    def test_task_text_cannot_include_old_state(self):
        with self.assertRaises(ValueError):frame(instruction='pick; simulation observation before action: target=1').checked()

    def test_falcon_depth_required(self):
        p=Inputs('falcon_esm',stats());p.push(frame(depth_z_m=None,intrinsics=None,extrinsics_OpenCV=None))
        with self.assertRaises(ValueError):p.sample()

    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):frame(depth_z_m=np.zeros((32,32))).checked()


class BankTests(unittest.TestCase):
    def bank(self,directory,split='TRAIN'):
        n=10;f=frame();path=Path(directory)
        np.savez(path/'episode.npz',raw_state21=np.stack([f.raw_state21]*n),
                 ee_position=np.stack([f.ee_position]*n),front_rgb=np.stack([f.front]*n),
                 wrist_rgb=np.stack([f.wrist]*n),canonical7=np.tile([.001,0,0,0,0,0,-1],(n,1)).astype(np.float32),
                 time_s=np.arange(n)*.1,depth_z_m=np.stack([f.depth_z_m]*n),
                 intrinsics=np.stack([f.intrinsics]*n),extrinsics_OpenCV=np.stack([f.extrinsics_OpenCV]*n))
        rows=[dict(episode_id='ep1',scene_id='train1',file='episode.npz',sha256=digest(path/'episode.npz'),
                   instruction=f.instruction,task='pick_place',source_kind='complete_original',target_name='red block')]
        (path/'index.json').write_text(json.dumps(dict(schema='vla21-episodes-v1',split=split,rows=rows)))
        return Bank(path/'index.json')

    def test_all_native_training_labels_without_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bank(d);s=b.statistics()
            for family in SPECS:
                with self.subTest(family=family):
                    x=b.sample(family,0,9,s)
                    self.assertTrue(x['action_is_pad'][1:].all())
                    if family=='xvla':self.assertEqual(x['action'].shape,(30,20))
                    elif family=='dita':self.assertEqual(x['normalized_action11'].shape,(11,7))
                    elif family=='seer':self.assertEqual(x['local_actions'].shape,(7,3,7))
                    else:self.assertEqual(x['action'].shape,(SPECS[family].horizon,7))

    def test_val_cannot_fit(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bank(d,'VAL')
            with self.assertRaises(ValueError):b.statistics()

    def test_scene_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bank(d)
            with self.assertRaises(ValueError):Bank(b.path,reserved_scenes=['train1'])

    def test_corrupted_payload_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bank(d);(Path(d)/'episode.npz').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):Bank(b.path)

    def test_train_stats_unique_timesteps(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bank(d);self.assertEqual(b.statistics()['robot21']['count'],10)
