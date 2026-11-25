"""Causal67 + native Qwen RGB + native EEF delta7, horizon50/padded32."""
import sys
import numpy as np
from VLA.intern_r72.common import SOURCE, BASE
sys.path.insert(0, str(SOURCE / 'src'))
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor
from lerobot.policies.internvla_a1_5.action_tokens import ensure_qwen35_action_tokens
from lerobot.transforms.utils import resize_with_pad
from lerobot.transforms.constants import SYSTEM_MESSAGE
from VLA.flower_r71.sampler import GraspData
from VLA.family_transfer_r54.data import action_stats


class Data(GraspData):
    def __init__(self):
        super().__init__(); self.horizon = 50
        self.intern_action_stats = action_stats(self, closure01=False)


def measured_feedback(raw):
    # Sensor facts, not advice, desired poses/actions, teacher phase or success.
    attachment = ('wrong object attached' if raw[30] else 'requested object attached') if raw[31] else 'no object attached'
    return (f'Measured feedback: {attachment}; target contact right={int(raw[28])}, left={int(raw[29])}; '
        f'measured closure={raw[6]:.3f}; jaw gap={raw[27]:.4f} m; '
        f'target minus gripper xyz={raw[11]:.4f},{raw[12]:.4f},{raw[13]:.4f} m; '
        f'goal minus object xyz={raw[14]:.4f},{raw[15]:.4f},{raw[16]:.4f} m; control frequency=10 Hz')


def targets(canonical, pad, stats):
    a, pad = np.asarray(canonical, np.float32), np.asarray(pad, bool)
    if a.shape != (50, 7) or pad.shape != (50,) or not np.isfinite(a).all():
        raise ValueError('Native50x7 finite real labels + explicit temporal pad required')
    mean, std = (np.asarray(stats[k], np.float32) for k in ('mean', 'std'))
    result = np.zeros((50, 32), np.float32)
    result[:, :7] = (a-mean)/(std+1e-6)
    # Do not fabricate a continuing nonzero terminal action. Padding does not
    # become a model attention/observation signal or FAST target supervision.
    result[pad] = 0
    return result


def decode(actions, stats):
    a = np.asarray(actions, np.float32)
    if a.ndim != 2 or a.shape[1] != 32 or not np.isfinite(a).all():
        raise ValueError('Native action expert must emit finite Hx32')
    mean, std = (np.asarray(stats[k], np.float32) for k in ('mean', 'std'))
    return a[:, :7]*(std+1e-6)+mean


class Inputs:
    def __init__(self, stats):
        self.stats = stats
        self.processor = Qwen3VLProcessor.from_pretrained(BASE / 'qwen_assets', local_files_only=True)
        ensure_qwen35_action_tokens(self.processor.tokenizer)
        self.fast = AutoProcessor.from_pretrained(BASE / 'fast_assets', trust_remote_code=True,
            local_files_only=True, tokenizer_file=str(BASE / 'fast_assets/tokenizer.json'))

    def __call__(self, sample, *, training=False, device='cpu'):
        raw = sample['raw64']; num = np.asarray(sample['numeric67'], np.float32)
        st = self.stats['scene67']; mean, std = (np.asarray(st[k], np.float32) for k in ('mean','std'))
        continuous = (num-mean)/(std+1e-8)
        if continuous.shape != (67,) or not np.isfinite(continuous).all(): raise ValueError('Full finite67 required')
        # Exact native discretization is retained as a redundant prefix. The
        # extra continuous state token prevents these bins losing precision.
        bins = np.digitize(continuous/3., np.linspace(-1,1,257)[:-1])-1
        text = ('Task: ' + sample['instruction'] + '; ' + measured_feedback(raw) +
            '; Control Mode: <end_effector>; State: ' + ' '.join(map(str,bins)) + '; Output: <Action>')
        images = [resize_with_pad(torch.from_numpy(sample[k].copy()).permute(2,0,1).float()/255.,224,224)
                  for k in ('front','wrist','map')]
        messages = [{'role':'system','content':SYSTEM_MESSAGE}, {'role':'user','content':[
            *[{'type':'image'} for _ in images], {'type':'text','text':text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=[prompt], images=images, do_rescale=False, return_tensors='pt',
            padding=False, truncation=False)
        prompt_ids = encoded.input_ids[0].tolist()
        valid_steps = int((~np.asarray(sample['action_is_pad'],bool)).sum()) if training else 0
        has_fast = training and valid_steps > 0
        if has_fast:
            action = targets(sample['action'], sample['action_is_pad'], self.stats['intern_action'])
            # The underlying official DCT processor supports variable length.
            # Supervise ONLY existing future actions, rather than fabricating
            # a repeated motion after a terminated demonstration. This is an
            # explicitly declared tail adaptation, not stock50-token labels.
            ids = self.fast(action[None,:valid_steps])[0]
            if not ids or min(ids)<0 or max(ids)>2047: raise ValueError('FAST codebook outside native2048 range')
            token_text = ''.join(f'<robot_action_{int(i)}>' for i in ids)
            # Use the identical actual prompt as inference. Labels are suffix
            # only, and the native model blocks them from action attention.
            full_text = prompt + token_text + self.processor.tokenizer.eos_token
            encoded = self.processor(text=[full_text], images=images, do_rescale=False, return_tensors='pt',
                padding=False, truncation=False)
            if encoded.input_ids[0,:len(prompt_ids)].tolist() != prompt_ids:
                raise ValueError('Train/infer actual prompt prefix differs')
        if encoded.input_ids.shape[1] > 1024: raise ValueError('Explicit extended1024 budget exceeded; no silent truncation')
        fast_mask = (encoded.input_ids >= 248077) & (encoded.input_ids <= 250124)
        # Include EOS in blocked suffix after the first ground-truth FAST ID.
        if has_fast:
            first = int(torch.nonzero(fast_mask[0])[0]); fast_mask[0,first:] = True
            labels = encoded.input_ids.clone(); labels[:,:len(prompt_ids)] = -100
        else: labels = None
        self.last_receipt = {'numeric': [1,67], 'native_horizon':50, 'native_action_dim':32,
            'images':3, 'image_grid_thw':encoded.image_grid_thw.tolist(),
            'pixel_values':list(encoded.pixel_values.shape), 'language_length':encoded.input_ids.shape[1],
            'prompt_length':len(prompt_ids), 'FAST_label_tokens':int(fast_mask.sum()),
            'FAST_supervised_real_horizon':valid_steps,
            'has_teacher_subtask':False, 'future_or_phase_in_observation':False,
            'continuous67_and_redundant_native_state_bins':True, 'no_truncation':True}
        return {'pixel_values':encoded.pixel_values.to(device), 'image_grid_thw':encoded.image_grid_thw.to(device),
            'lang_tokens':encoded.input_ids.to(device), 'lang_masks':encoded.attention_mask.to(device),
            'state':torch.from_numpy(continuous)[None].to(device),
            'fast_token_mask':fast_mask.to(device), 'labels':None if labels is None else labels.to(device)}
