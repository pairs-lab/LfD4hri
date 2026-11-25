"""Falsifiable CPU contracts, sample visual review, and disjoint prior split proof."""
import unittest
import numpy as np
from .data import normalize_actions, denormalize_actions, observation74, sample

class Contracts(unittest.TestCase):
    def setUp(self):
        self.stats = {'actions':{'q01':[-.05]*6+[-1.], 'q99':[.05]*6+[1.]},
                      'scene67':{'mean':[0.]*67,'std':[1.]*67}}

    def test_action_roundtrip_and_signed_grip(self):
        a = np.random.default_rng(84).uniform(-.04,.04,(8,7)).astype(np.float32)
        a[:,6] = [-1,1,-1,1,-1,1,-1,1]
        np.testing.assert_allclose(denormalize_actions(normalize_actions(a,self.stats),self.stats),a,atol=1e-8)

    def test_no_target_controller(self):
        # Decoding has no environment, target, desired EEF, or expert argument.
        import inspect
        self.assertEqual(list(inspect.signature(denormalize_actions).parameters),['a','stats'])

def main():
    from collections import Counter
    from PIL import Image, ImageDraw
    from VLA.common.artifact_io import atomic_json
    from VLA.common.provenance import sha256_file
    from VLA.family_transfer_r54.data import anchors, phase_anchors
    from .data import Data, statistics
    from .common import ROOT,OUT,REPO,TASKS,read,utc
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise ValueError('Numerical action interface failed')
    prior = ROOT/'VLA/audits/autonomous_vla_r83_20261020'
    if not read(prior/'CPU_AUDIT.json')['passed']:
        raise ValueError('Complete TRAIN/split audit missing')
    data = Data()
    stats = statistics(data)
    selected = anchors(data)
    canvas = Image.new('RGB',(768,280*len(selected)),'white')
    draw = ImageDraw.Draw(canvas)
    for row,(ei,fi) in enumerate(selected):
        s = sample(data,ei,fi,stats)
        if s['history74'].shape != (8,74) or not np.isfinite(s['history74']).all():
            raise ValueError('Missing complete causal feedback')
        if s['history_source_indices'].max() != fi or s['history_source_indices'].min() < 0:
            raise ValueError('Future/invalid observation history')
        n = data.numeric(ei)
        np.testing.assert_allclose(s['history74'][-1,:6], n['native8'][fi,:6])
        if s['history74'][-1,6] != n['scene64'][fi,6]:
            raise ValueError('Requested rather than measured closure supplied')
        pad = s['action_is_pad']
        a = denormalize_actions(s['normalized_action'],stats)
        if pad.any():
            np.testing.assert_allclose(a[pad,:6],0,atol=1e-7)
        draw.text((4,row*280+2),f"{s['task']} episode{ei} frame{fi}; native74 x history8",fill='black')
        for col,name in enumerate(('front','wrist','map')):
            canvas.paste(Image.fromarray(s[name]).resize((256,256)),(col*256,row*280+22))
    canvas.save(OUT/'VLANEXT_PREPROCESS_REVIEW.png')
    phase_ids, events = phase_anchors(data)
    for task in TASKS:
        ids = [(ei,fi) for ei,fi in phase_ids if data.rows[ei]['task']==task]
        sheet = Image.new('RGB',(768,280*len(ids)),'white')
        painter = ImageDraw.Draw(sheet)
        for row,(ei,fi) in enumerate(ids):
            s = sample(data,ei,fi,stats)
            painter.text((3,row*280+2),f"{task} ep{ei}/t{fi}: {','.join(events[ei,fi])}",fill='black')
            for col,name in enumerate(('front','wrist','map')):
                sheet.paste(Image.fromarray(s[name]).resize((256,256)),(col*256,row*280+22))
        sheet.save(OUT/(task+'_PHASE_REVIEW.png'))
    atomic_json(OUT/'VLANEXT_DATA_AUDIT.json', {'passed_contracts':True,'UTC':utc(),
        'TRAIN_trajectories':len(data.rows),'TRAIN_frames':len(data),
        'per_action_frames':dict(Counter({task:sum(r['frames'] for r in data.rows if r['task']==task) for task in TASKS})),
        'per_action_trajectories':dict(Counter(r['task'] for r in data.rows)),
        'eight_review_anchors':selected, 'stats':stats,
        'additional_phase_anchors':phase_ids,
        'existing_split_proof':str(prior/'GEOMETRY_EXCLUSION.json'),
        'existing_split_proof_sha256':sha256_file(prior/'GEOMETRY_EXCLUSION.json'),
        'original_images_and_native_FK_labels_unchanged':True,
        'expert_data_NOT_proven_TD3_generated':True,
        'wrong_object_drop_recovery_demonstrations_NOT_certified_complete':True,
        'native_recipe_config':str(REPO/'config/VLANeXt-S_config.yaml'),
        'selected_foundation_is_VLM_not_robot_action_pretrained':True,
        'native_loss':'official flow velocity MSE + DCT0.1, lowfreq5/highfreq0 split0.125',
        'official_tail_padding':'zero motion with final signed-close gripper; NOT repeated nonzero motion',
        'declared_changes':['same PyBullet world/local action units with TRAIN normalization',
            'native proprio7 prefix + bounded causal67 appended to linear VLM projector',
            'front/wrist + privileged map third visual view, not RGB-D camera',
            'robot FK and history from our actual same robot',
            'SDPA instead of FlashAttention2; shared existing torch2.11 instead of2.4',
            'RGB photometric/crop augmentation as source preset; privileged map unaugmented'],
        'data_ready_does_not_prove_closed_loop_success':True})

if __name__ == '__main__':
    main()
