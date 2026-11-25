"""Real pretrained X-VLA inference/backprop gate; not task SR or a long fit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.privileged_inputs import append_numeric67, expand_xvla_proprio, inject_native_lora

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
CP = ROOT/'VLA/checkpoints/xvla-base'
FIXTURE = BASE/'native_TRAIN_action_probe_r37_v2'


def main(args):
    from lerobot.configs import PolicyFeature, FeatureType
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = 'native_checkpoint_loading_CPU'

    def save(status='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': status, 'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_SR_established': False, 'long_finetune_qualified': False, **extra})

    try:
        save()
        torch.set_num_threads(2)
        cfg = XVLAConfig.from_pretrained(str(CP))
        cfg.device = 'cpu'
        cfg.dtype = torch.float32
        model = XVLAPolicy.from_pretrained(CP, config=cfg)
        count = sum(p.numel() for p in model.parameters())
        # Native strict load above has completed before changing input width.
        extension = expand_xvla_proprio(model.model.transformer, cfg.max_state_dim)
        cfg.max_state_dim = extension['new_proprio_dim']
        cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(cfg.max_state_dim,))
        model.model.dim_proprio = cfg.max_state_dim
        for p in model.parameters():
            p.requires_grad_(False)
        if args.lora_rank:
            model = inject_native_lora(model, args.lora_rank)
        # Native soft prompts and per-domain action input/output layers. No
        # auxiliary planner/action residual replaces the VLA transformer.
        for name in ('soft_prompt_hub', 'action_encoder', 'action_decoder'):
            for p in getattr(model.model.transformer, name).parameters():
                p.requires_grad_(True)
        trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
        if any(p.dtype != torch.float32 for p in trainable.values()):
            raise ValueError('Trainable master weights must remain FP32')
        atomic_json(out/'native_load.json', {'passed_strict_native_load': True,
            'base_checkpoint': str(CP), 'original_parameters': count,
            'parameters_after_input_extension': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in trainable.values()),
            'all_linear_lora_rank': args.lora_rank,
            'extension': extension, 'own_finetune_NOT_completed': True,
            'source_model_revision': 'cdb7964e4fe842935d671bfab5a5ebe00a96648c'})
        # Native checkpoint refers to BART's tokenizer but does not bundle it.
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large',
            cache_dir=str(ROOT/'VLA/checkpoints/huggingface/hub'))
        fixture_proof = json.loads((FIXTURE/'audit.json').read_text())
        fixtures = json.loads((FIXTURE/'requests_with_LOCAL_gold.json').read_text())
        if (not fixture_proof['passed'] or sha256_file(FIXTURE/'requests_with_LOCAL_gold.json')
                != fixture_proof['request_file_sha256']):
            raise ValueError('Actual TRAIN fixture identity changed')
        item = fixtures[0]
        path = Path(item['source_file'])
        if sha256_file(path) != fixture_proof['bound_files'][str(path)]:
            raise ValueError('Actual TRAIN source bytes changed')
        frame = item['source_frame']
        with np.load(path, allow_pickle=False) as data:
            pose = np.r_[data['ee_position'][frame], data['ee_rotation_6d'][frame],
                         np.float32(1-data['gripper_open'][frame])].astype(np.float32)
            prompt = str(data['prompt'].item())
            images = [data[key][frame].copy() for key in ('front_rgb', 'wrist_rgb')]
        prefix = np.r_[pose, np.zeros(3), [1, 0, 0, 0, 1, 0], 0].astype(np.float32)
        stats = json.loads((BASE/'all_family_privileged_r33_v4/rdt_fit_2000/checkpoints/step-002000/pybullet_rdt_config.json').read_text())['scene67']['stats']
        state = append_numeric67(prefix, item['payload']['scene64'], stats)
        tokenized = tokenizer(prompt, return_tensors='pt', padding='max_length',
                              max_length=50, truncation=True)
        if len(tokenizer.encode(prompt)) > 50:
            raise ValueError('Task instruction would be truncated')
        batch = {OBS_STATE: torch.from_numpy(state).unsqueeze(0),
                 OBS_LANGUAGE_TOKENS: tokenized['input_ids']}
        # Actual native ImageNet normalization; model's own resize_with_pad
        # performs its checkpoint-defined224 resize, no unrelated LIBERO flip.
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        for role, pixels in zip(('image', 'image2'), images, strict=True):
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).float()/255
            batch['observation.images.'+role] = ((tensor-mean)/std).unsqueeze(0)
        for role, pixels in zip(('front', 'wrist'), images, strict=True):
            Image.fromarray(pixels).save(out/(role+'_actual_TRAIN_rgb.png'))
        # Use real source command for the XYZ part. Full robot/FK parity and
        # complete-source batching still required before any long fit.
        target = prefix.copy()
        target[:3] += np.asarray(item['canonical_gold_LOCAL_ONLY'][:3], np.float32)
        target[9] = (float(item['canonical_gold_LOCAL_ONLY'][6])+1)/2
        batch['action'] = torch.from_numpy(np.tile(target, (cfg.chunk_size, 1))).unsqueeze(0)
        stage = 'actual_CUDA_native_prediction_and_gradient_gate'
        save()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(12000*2**20/total)
        model = model.to('cuda')
        cfg.device = 'cuda'
        batch = {k: v.to('cuda') for k, v in batch.items()}
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        def predict():
            with torch.random.fork_rng(devices=[0]), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                torch.manual_seed(202643)
                return model.predict_action_chunk(batch).float().cpu()
        first = predict()
        second = predict()
        if first.shape != (1, cfg.chunk_size, 20) or not torch.isfinite(first).all():
            raise ValueError('Malformed native X-VLA prediction')
        difference = float((first-second).abs().max())
        if difference != 0:
            raise ValueError('Identical native request was not exactly repeatable')
        model.train()
        model.model.vlm.eval()
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = model(batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native X-VLA loss')
        loss.backward()
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing/nonfinite actual native gradients')
        feedback_gradient = model.model.transformer.action_encoder.fc.weight.grad.reshape(
            cfg.num_domains, -1, cfg.hidden_size)[:, 40:107]
        if not torch.count_nonzero(feedback_gradient):
            raise ValueError('Appended privileged information has no gradient path')
        atomic_json(out/'actual_CUDA_gate.json', {'passed': True,
            'scope': 'actual_pretrained_native_CUDA_prediction_gradient_NOT_model_SR_or_fit_qualification',
            'prediction_shape': list(first.shape), 'repeat_max_abs_difference': difference,
            'loss': float(loss.detach()), 'native_losses': metrics,
            'trainable_master_dtype': 'float32', 'compute': 'bfloat16_autocast',
            'feedback_gradient_nonzero_values': int(torch.count_nonzero(feedback_gradient)),
            'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
            'forward_backward_seconds': time.monotonic()-started,
            'source_frame': frame, 'TRAIN_fixture_only': True, 'state_dim': len(state),
            'task_or_feedback_truncated': False, 'action_labels_are_partial_gate_fixture_only': True,
            'optimizer_updates': 0, 'task_SR_established': False,
            'complete_source_FK_parity_own_optimizer_save_reload_and_robot8_still_required': True})
        stage = 'native_CUDA_gradient_gate_passed_NOT_long_fit_or_task_SR'
        save('completed_native_CUDA_gate')
    except Exception as error:
        save('failed_preserve_actual_evidence', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lora-rank', type=int, choices=(0, 8), default=8)
    main(parser.parse_args())
