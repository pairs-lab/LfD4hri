"""Declared native-policy repairs, not a privileged action planner."""
from pathlib import Path
import numpy as np
from VLA.autonomy_r83.common import read, verify


def binary_evo_gripper(actions):
    result = np.asarray(actions, np.float32).copy()
    # Official Evo LIBERO client uses OPEN01 >0.5. R80 returned continuous
    # signed-close, so this corresponds to negative -> open, otherwise close.
    result[:, 6] = np.where(result[:, 6] < 0, -1., 1.)
    return result


class Runtime:
    def __init__(self, family, parent, stats=None, variant='map3', *, checkpoint=None):
        import torch
        self.family, self.variant, self.parent = family, variant, Path(parent)
        self.stats = stats
        self.last_receipt = {}
        if checkpoint:
            meta = read(Path(checkpoint) / 'ADAPTATION.json')
            verify(meta['code'])
            self.stats = meta['stats']
            self.parent = Path(meta['parent'])
        if family == 'evo1':
            from VLA.evo_transfer_r80.native import Runtime as Native
            self.stats = read(self.parent / 'stats.json')
            self.native = Native(self.stats, self.parent)
            self.horizon = 50
        elif family == 'smolvla':
            from VLA.recovery_r56.entry import register_smol
            from VLA.family_transfer_r54.smol import Runtime as Native, BaselineRuntime
            register_smol()
            if variant == 'baseline':
                self.native = BaselineRuntime(self.parent)
            else:
                loaded = Path(checkpoint) / 'native_model' if checkpoint else self.parent
                self.native = Native(loaded, self.stats)
            self.horizon = 5
        elif family == 'vla_adapter':
            from VLA.new_vlas.runtime import create_runtime as two_view
            from VLA.privileged_visual_r53.runtime import create_runtime as three_view
            self.native = (two_view if variant == 'baseline' else three_view)(family, self.stats)
            path = Path(checkpoint) / 'trainables.pt' if checkpoint else self.parent / 'trainable_masters.pt'
            self.native.load_trainables(path)
            self.native.to('cuda')
            self.horizon = 8
        else:
            raise ValueError('No unimplemented family may enter automation: ' + family)
        if family == 'smolvla':
            self.params = {n: p for n, p in self.native.model.named_parameters() if p.requires_grad}
        elif family == 'vla_adapter':
            self.params = self.native.parameters()
        else:
            self.params = self.native.trainable
        if any(p.dtype != torch.float32 for p in self.params.values()):
            raise ValueError('Learned masters must remain FP32')

    def sample(self, data, ei, fi):
        sample = data.at(ei, fi)
        if self.family != 'vla_adapter':
            return sample
        from VLA.new_vlas.action_contracts import normalize_bounds
        from VLA.stable_native_r44.conditioning import bounded_append
        n = data.numeric(ei)
        native = data.sample(ei, fi, 'vla_adapter')
        prefix = normalize_bounds(n['native8'][fi], self.stats['adapter']['proprio'])
        native['state'] = bounded_append(prefix, n['scene64'][fi], self.stats['scene67'])
        native['privileged_map'] = sample['map']
        native['action_is_pad'] = fi + np.arange(8) >= data.rows[ei]['frames']
        return native

    def loss(self, sample):
        import torch
        if self.family == 'smolvla':
            self.native.model.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return self.native.loss(sample)
        if self.family != 'vla_adapter':
            raise ValueError('No blind repeated Evo training in this bounded recipe')
        # Same official train-time head/features/L1, but mask fake tail labels.
        r = self.native
        r.train()
        r.training_capture, r.captured = True, None
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                r.model.predict_action(**r.batch(sample), unnorm_key='pybullet_train',
                    proprio=sample['state'], proprio_projector=r.projector, action_head=r.head)
                prediction = r.captured
                target = torch.from_numpy(sample['action'])[None].cuda()
                mask = torch.from_numpy(~sample['action_is_pad'])[None, :, None].cuda()
                if prediction is None or not prediction.requires_grad or prediction.shape != target.shape:
                    raise ValueError('Detached or malformed official Adapter train-time head')
                return ((prediction.float() - target).abs() * mask).sum() / (mask.sum() * 7)
        finally:
            r.training_capture, r.captured = False, None

    def observed(self, payload):
        from VLA.privileged_native_server import decode_image
        from VLA.scene64_conditioning import encode_numeric, validate_scene
        from VLA.privileged_visual_r53.scene_map import render, task_from_instruction
        raw = validate_scene(payload['scene64'])
        task = task_from_instruction(payload['instruction'])
        sample = {'front': decode_image(payload['front_png']), 'wrist': decode_image(payload['wrist_png']),
                  'map': render(raw, task), 'raw64': raw, 'numeric67': encode_numeric(raw),
                  'instruction': payload['instruction'], 'task': task}
        if self.family == 'evo1':
            from VLA.new_vlas.action_contracts import native8
            sample['native8'] = native8(np.asarray(payload['ee_position'], np.float32),
                                       np.asarray(payload['ee_rotation_6d'], np.float32), raw[27])
        elif self.family == 'vla_adapter':
            from VLA.new_vlas.action_contracts import native8, normalize_bounds
            from VLA.stable_native_r44.conditioning import bounded_append
            prefix = normalize_bounds(native8(np.asarray(payload['ee_position'], np.float32),
                np.asarray(payload['ee_rotation_6d'], np.float32), raw[27]), self.stats['adapter']['proprio'])
            sample['state'] = bounded_append(prefix, raw, self.stats['scene67'])
            sample['privileged_map'] = sample['map']
        return sample

    def predict(self, sample, seed):
        import torch
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(int(seed))
            if self.family in ('evo1', 'smolvla'):
                actions = self.native.predict(sample, seed)
                if self.family == 'evo1' and self.variant == 'binary':
                    actions = binary_evo_gripper(actions)
            else:
                from VLA.new_vlas.policy_outputs import native_gripper_command
                actions = np.asarray(self.native.predict(sample), np.float32).copy()
                actions[:, 6] = [native_gripper_command(v, convention='adapter_open01') for v in actions[:, 6]]
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError('Malformed actual native canonical actions')
        self.last_receipt = {'family': self.family, 'variant': self.variant,
            'actual_native_views': 2 if self.variant == 'baseline' else 3,
            'numeric_dim': 67, 'future_teacher_phase_or_labels': False,
            'geometric_action_residual_or_Oracle': False}
        if self.family == 'evo1':
            self.last_receipt['native'] = self.native.last_input
        elif self.family == 'vla_adapter' and self.variant != 'baseline':
            receipt = self.native.r53_native_visual_receipt
            if receipt['actual_views'] != 3:
                raise ValueError('Missing native third visual input')
            self.last_receipt['native'] = receipt
        return actions

    def save_weights(self, out):
        from safetensors.torch import save_file
        out = Path(out)
        if self.family == 'smolvla':
            self.native.model.save_pretrained(out / 'native_model')
            path = out / 'native_model/model.safetensors'
        else:
            path = out / 'trainables.pt'
            self.native.save_trainables(path)
        return path
