"""Real native VLA-Adapter-Pro checkpoint/LoRA gradient and memory gate.

Uses a predeclared real TRAIN frame; no simulator controls, SR claims, or
long-fit admission. Native LIBERO normalization is NOT a PyBullet benchmark.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.new_vlas.adapter_loading import import_native, install_amp_query_cast
from VLA.new_vlas.privileged_inputs import append_numeric67, expand_adapter_proprio

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
CP = ROOT/'VLA/checkpoints/vla-adapter-spatial-pro'


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = 'waiting_existing_new_model_gate'
    def save(status='running', **extra):
        atomic_json(out/'status.json', {'pid': os.getpid(), 'status': status, 'stage': stage,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'long_fit_qualified': False, 'task_SR_established': False, **extra})
    try:
        save()
        if args.wait_for:
            until = time.monotonic()+3600
            while True:
                status = json.loads((args.wait_for/'status.json').read_text())
                if status['status'] != 'running':
                    break
                if time.monotonic() > until:
                    raise TimeoutError('Existing new-model native gate did not finish')
                time.sleep(5)
        from VLA.run_cross_vla120 import gpu_state
        until = time.monotonic()+3600
        while True:
            gpu = gpu_state()
            if not gpu['compute_pids'] and gpu['free_mib'] >= 14000:
                break
            if time.monotonic() > until:
                raise TimeoutError('No safe exclusive GPU slot for native gate')
            save()
            time.sleep(5)
        stage = 'native_strict_checkpoint_and_action_head_load'
        save()
        torch.set_num_threads(2)
        Config, Model, ImageProcessor, Processor, Head, Projector = import_native()
        cfg = Config.from_pretrained(CP)
        model, load_info = Model.from_pretrained(CP, config=cfg, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation='eager', output_loading_info=True)
        if load_info['missing_keys'] or load_info['unexpected_keys'] or load_info.get('mismatched_keys'):
            raise ValueError('Native base checkpoint/source mismatch: ' + str(load_info))
        # Match the official checkpoint loader: the fine-tuned statistics file
        # replaces the base config's OXE mixture statistics (not merged with it).
        model.norm_stats = json.loads((CP/'dataset_statistics.json').read_text())
        if 'libero_spatial_no_noops' not in model.norm_stats:
            raise ValueError('Official fine-tuned dataset statistics are missing')
        model.vision_backbone.set_num_images_in_input(2)
        head = Head(input_dim=model.llm_dim, hidden_dim=model.llm_dim,
                    action_dim=7, num_task_tokens=512, use_pro_version=True)
        projector = Projector(llm_dim=model.llm_dim, proprio_dim=8)
        for module, filename in ((head, 'action_head--checkpoint.pt'),
                                 (projector, 'proprio_projector--checkpoint.pt')):
            saved = torch.load(CP/filename, map_location='cpu', weights_only=True, mmap=True)
            values = {key.removeprefix('module.'): value for key, value in saved.items()}
            module.load_state_dict(values, strict=True)
            if any(not torch.equal(module.state_dict()[key], value.float()) for key, value in values.items()):
                raise ValueError('Native saved head/projector values changed on FP32 master load')
        extension = expand_adapter_proprio(projector)
        original_parameters = sum(p.numel() for p in model.parameters())
        amp_boundary = install_amp_query_cast(model)
        from peft import get_peft_model, LoraConfig
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.,
            target_modules='all-linear', init_lora_weights='gaussian'))
        for name, parameter in model.named_parameters():
            if 'action_queries' in name:
                parameter.requires_grad_(True)
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        learnable = [p for m in (model, head, projector) for p in m.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in learnable):
            raise ValueError('Trainable masters must remain FP32')
        atomic_json(out/'native_load.json', {'passed_strict_load': True,
            'base_parameters': original_parameters,
            'head_parameters': sum(p.numel() for p in head.parameters()),
            'projector_parameters_after_extension': sum(p.numel() for p in projector.parameters()),
            'trainable_parameters': sum(p.numel() for p in learnable),
            'LoRA_rank': 64, 'extension': extension, 'head_and_projector_exact_saved_values': True,
            'amp_boundary': amp_boundary,
            'source_revision': 'cb023c6771fc62ea729d218f09ac0583c14f59d4'})
        fixture = BASE/'native_TRAIN_action_probe_r37_v2'
        proof = json.loads((fixture/'audit.json').read_text())
        requests_file = fixture/'requests_with_LOCAL_gold.json'
        if not proof['passed'] or sha256_file(requests_file) != proof['request_file_sha256']:
            raise ValueError('Actual TRAIN fixture changed')
        item = json.loads(requests_file.read_text())[0]
        path = Path(item['source_file'])
        if sha256_file(path) != proof['bound_files'][str(path)]:
            raise ValueError('TRAIN source bytes changed')
        with np.load(path, allow_pickle=False) as data:
            frame = item['source_frame']
            xyz = data['ee_position'][frame].copy()
            rotation = data['ee_rotation_6d'][frame].copy()
            images = [Image.fromarray(data[k][frame]) for k in ('front_rgb', 'wrist_rgb')]
        from scipy.spatial.transform import Rotation
        columns = rotation.reshape(2, 3).T
        matrix = np.c_[columns, np.cross(columns[:, 0], columns[:, 1])]
        raw = np.asarray(item['payload']['scene64'], np.float32)
        prefix = np.r_[xyz, Rotation.from_matrix(matrix).as_rotvec(), raw[27]/2, -raw[27]/2]
        normalizer = model.norm_stats['libero_spatial_no_noops']['proprio']
        low, high = [np.asarray(normalizer[key], np.float32) for key in ('q01', 'q99')]
        prefix = np.clip(2*(prefix-low)/(high-low+1e-8)-1, -1, 1)
        stats = json.loads((BASE/'all_family_privileged_r33_v4/rdt_fit_2000/checkpoints/step-002000/pybullet_rdt_config.json').read_text())['scene67']['stats']
        state = append_numeric67(prefix, raw, stats)
        instruction = item['payload']['instruction'].lower()
        prompt = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                  '<|im_start|>user\nWhat action should the robot take to '+instruction+
                  '?<|im_end|>\n<|im_start|>assistant\n')
        processor = Processor(ImageProcessor.from_pretrained(CP), AutoTokenizer.from_pretrained(CP))
        inputs = processor(prompt, images[0], return_tensors='pt')
        wrist = processor(prompt, images[1], return_tensors='pt')
        inputs['pixel_values'] = torch.cat((inputs['pixel_values'], wrist['pixel_values']), dim=1)
        for role, im in zip(('front', 'wrist'), images, strict=True):
            im.save(out/(role+'_actual_TRAIN_rgb.png'))
        stage = 'actual_native_LoRA64_prediction_and_gradient_CUDA_gate'
        save()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(14000*2**20/total)
        model, head, projector = [m.to('cuda') for m in (model, head, projector)]
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        captured = {}
        original_predict = head.predict_action
        def capture(self, features, **kwargs):
            captured['features'] = features
            return original_predict(features, **kwargs)
        head.predict_action = MethodType(capture, head)
        def predict():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model.predict_action(**inputs, unnorm_key='libero_spatial_no_noops',
                    proprio=state, proprio_projector=projector, action_head=head)[0]
        for module in (model, head, projector):
            module.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            first, again = predict(), predict()
        if first.shape != (8, 7) or not np.isfinite(first).all() or not np.array_equal(first, again):
            raise ValueError('Native prediction shape/finite/repeatability failed')
        for module in (model, head, projector):
            module.train()
        started = time.monotonic()
        predict()
        features = captured['features']
        if not features.requires_grad:
            raise ValueError('Native VLA forward detached the real LoRA graph')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = original_predict(features, proprio=torch.as_tensor(state, device='cuda').unsqueeze(0),
                proprio_projector=projector, phase='Training')
            # Partial real-frame label is ONLY a numerical/backprop fixture,
            # not native full-source action normalization or fit admission.
            label = torch.tensor(item['canonical_gold_LOCAL_ONLY'], device='cuda').repeat(1, 8, 1)
            loss = torch.nn.functional.l1_loss(output, label)
        loss.backward()
        gradients = [p.grad for p in learnable if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing or nonfinite native LoRA/head gradients')
        feedback = projector.fc1.weight.grad[:, 8:]
        if not torch.count_nonzero(feedback):
            raise ValueError('No privileged feedback gradient')
        atomic_json(out/'actual_CUDA_gate.json', {'passed': True,
            'scope': 'native_checkpoint_LoRA64_CUDA_gradient_NOT_finetune_qualification_or_SR',
            'prediction_shape': list(first.shape), 'identical_request_exact': True,
            'native_visual_channels': int(inputs['pixel_values'].shape[1]),
            'state_dim': len(state), 'full67_feedback_gradient_nonzero_values': int(torch.count_nonzero(feedback)),
            'trainable_master_dtype': 'float32', 'compute': 'bfloat16_autocast',
            'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
            'forward_backward_seconds': time.monotonic()-started, 'fixture_loss_NOT_quality': float(loss.detach()),
            'optimizer_updates': 0, 'native_LIBERO_normalization_NOT_PyBullet_action_contract': True,
            'partial_fixture_NOT_complete_source_labels': True, 'task_SR_established': False,
            'own_complete_source_action_contract_optimizer_save_reload_robot8_still_required': True})
        stage = 'actual_native_LoRA64_CUDA_gate_completed_NOT_fit_or_SR'
        save('completed_native_CUDA_gate')
    except Exception as error:
        save('failed_preserve_actual_evidence', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path)
    main(parser.parse_args())
