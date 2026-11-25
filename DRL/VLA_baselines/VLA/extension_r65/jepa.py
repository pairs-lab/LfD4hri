"""Native VLA-JEPA; explicit pretrained-state8→simulator67 adaptation.

No changes to native DiT, flow loss, Qwen features or neural action decoder.
The meta constructor avoids loading a redundant foundation checkpoint before
loading every actual Qwen value from the pretrained VLA safetensors file.
"""
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import BASE, QWEN, read


def config():
    import torch
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
    c = VLAJEPAConfig.from_pretrained(str(BASE))
    c.qwen_model_name = str(QWEN)
    c.freeze_qwen = True
    c.enable_world_model = False
    c.world_model_loss_weight = 0.
    c.binarize_gripper_action = False
    c.pre_snap_gripper_action = False
    c.clip_normalized_actions = False
    c.use_relative_actions = False  # Labels already canonical EEF deltas.
    c.resize_images_to = (224,224)
    c.device = 'cuda'; c.dtype = torch.bfloat16
    c.state_dim = 67; c.action_dim = 7
    c.n_obs_steps = 1; c.chunk_size = c.n_action_steps = 7
    c.input_features = {f'observation.images.{name}': PolicyFeature(type=FeatureType.VISUAL, shape=(3,384,384))
                        for name in ('front', 'wrist', 'simGT')}
    c.input_features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(67,))
    c.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    c.validate_features()
    return c


def stats(data):
    from VLA.family_transfer_r54.data import action_stats
    return {'state': data.stats['scene67'], 'action': action_stats(data)}


def batch(x, stat, device='cuda', train=False):
    import torch
    result = {f'observation.images.{k}': torch.from_numpy(x[src].copy()).permute(2,0,1)[None].to(device).float()/255
              for k,src in (('front','front'), ('wrist','wrist'), ('simGT','map'))}
    state = (x['numeric67'] - np.asarray(stat['state']['mean'],np.float32)) / np.asarray(stat['state']['std'],np.float32)
    result['observation.state'] = torch.from_numpy(np.clip(state,-5.,5.).copy())[None].to(device)
    result['task'] = [x['instruction']]
    if train:
        action = (x['action'] - np.asarray(stat['action']['mean'],np.float32))/np.asarray(stat['action']['std'],np.float32)
        result['action'] = torch.from_numpy(action.copy())[None].to(device)
        result['action_is_pad'] = torch.from_numpy(x['action_is_pad'].copy())[None].to(device)
    return result


def configure_fp32_action_loss(model):
    """Keep native flow formula; fix bf16 Qwen→FP32 native DiT boundary.

    Native _action_loss converts state to the Qwen hidden dtype. With learned
    FP32 masters this must use FP32 hidden features, just like native predict.
    No layers/loss terms/actions change, and original state precision is kept.
    """
    native=model._action_loss
    def fp32(embodied_action_tokens,actions,state,action_is_pad,reduction='mean'):
        return native(embodied_action_tokens.float(),actions.float(),
            None if state is None else state.float(),action_is_pad,reduction=reduction)
    model._action_loss=fp32


class Runtime:
    def __init__(self, stat, checkpoint=None):
        import torch
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
        self.stat = stat; self.cfg = config()
        original_class = Qwen3VLForConditionalGeneration
        def architecture_only(*_args, **_kwargs):
            return original_class(AutoConfig.from_pretrained(str(QWEN)))
        torch.manual_seed(202665)
        with patch.object(original_class, 'from_pretrained', architecture_only), init_empty_weights():
            self.model = VLAJEPAPolicy(self.cfg)
        current = self.model.state_dict(); values = {}; skipped = []; reset = []
        with safe_open(BASE/'model.safetensors', framework='pt', device='cpu') as saved:
            for key in saved.keys():
                if key.startswith(('model.video_encoder.', 'model.video_predictor.')):
                    skipped.append(key); continue
                if key not in current:
                    raise ValueError('Unaccounted pretrained tensor: '+key)
                if tuple(saved.get_slice(key).get_shape()) != tuple(current[key].shape):
                    if key != 'model.action_model.state_encoder.layer1.weight':
                        raise ValueError('Unqualified native pretrained shape mismatch: '+key)
                    reset.append(key); continue
                values[key] = saved.get_tensor(key)
        if reset != ['model.action_model.state_encoder.layer1.weight']:
            raise ValueError('Require explicit state8→67 projection migration')
        layer = torch.nn.Linear(67, self.cfg.action_hidden_size, dtype=torch.float32)
        values[reset[0]] = layer.weight.detach()
        missing, extra = self.model.load_state_dict(values, strict=False, assign=True)
        if missing or extra:
            raise ValueError(f'Unloaded native tensors: {missing}; unexpected: {extra}')
        self.model.model.qwen.requires_grad_(False)
        self.model.model.action_model.requires_grad_(True)
        self.model.model.action_model.float()  # Native learned masters and Adam FP32.
        configure_fp32_action_loss(self.model.model)
        if any(p.is_meta for p in self.model.parameters()):
            raise ValueError('A native learned/frozen value remains unmaterialized')
        self.model.to('cuda')
        self.trainable = {n:p for n,p in self.model.named_parameters() if p.requires_grad}
        if not self.trainable or any(not n.startswith('model.action_model.') for n in self.trainable):
            raise ValueError('Wrong native action-head-only learning scope')
        # Observe the actual native Qwen/DiT boundaries, not just our payload.
        self._seen_inputs={};self._expected_state=None
        native_inputs=self.model.model.qwen.build_inputs
        def checked_inputs(*args,**kwargs):
            prepared=native_inputs(*args,**kwargs)
            grid=prepared['image_grid_thw']
            if grid.ndim!=2 or grid.shape[0]!=3:
                raise ValueError('Native Qwen did not consume front/wrist/GT3')
            self._seen_inputs['actual_Qwen_image_grid_thw']=grid.detach().cpu().tolist()
            return prepared
        self.model.model.qwen.build_inputs=checked_inputs
        def checked_state(_module,args):
            value=args[0]
            if value.shape[-1]!=67 or self._expected_state is None:
                raise ValueError('Native DiT omitted full privileged state67')
            expected=self._expected_state.reshape(1,67).to(value.dtype)
            if not torch.equal(value.reshape(-1,67),expected.expand(value.numel()//67,67)):
                raise ValueError('Actual native DiT privileged values differ')
            self._seen_inputs['actual_DiT_state_dimensions']=67
        self.model.model.action_model.state_encoder.register_forward_pre_hook(checked_state)
        self.load_receipt = {'all_retained_pretrained_tensors_loaded': True,
            'reinitialized': reset, 'world_auxiliary_tensors_omitted': len(skipped),
            'trainable_parameters': sum(p.numel() for p in self.trainable.values()),
            'pretrained_VLA_Qwen_frozen_not_randomized': True}
        if checkpoint is not None:
            from safetensors.torch import load_file
            saved = load_file(str(Path(checkpoint)/'learned.safetensors'), device='cpu')
            if set(saved) != set(self.trainable):
                raise ValueError('Learned native tensor names differ on reload')
            with torch.no_grad():
                for name, value in saved.items():
                    if value.dtype != torch.float32 or value.shape != self.trainable[name].shape:
                        raise ValueError('Lossy/reshaped learned native value')
                    self.trainable[name].copy_(value)
                    if not torch.equal(self.trainable[name].detach().cpu(), value):
                        raise ValueError('Saved learned value did not reload exactly')

    def loss(self, x):
        self.model.train(); self.model.model.qwen.eval()
        prepared=batch(x,self.stat,train=True);self._begin_inputs(prepared)
        result=self.model(prepared)[0];self._finish_inputs()
        return result

    def _begin_inputs(self,prepared):
        self._seen_inputs={};self._expected_state=prepared['observation.state'].detach()

    def _finish_inputs(self):
        if (not self._seen_inputs.get('actual_Qwen_image_grid_thw') or
            self._seen_inputs.get('actual_DiT_state_dimensions')!=67):
            raise ValueError('Actual native VLA privileged-input receipt incomplete')
        self.last_input_receipt=dict(self._seen_inputs)

    def predict(self, x, seed):
        import torch
        self.model.eval(); self.model.reset()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(int(seed))
            prepared=batch(x,self.stat);self._begin_inputs(prepared)
            native = self.model.predict_action_chunk(prepared).float().cpu().numpy()[0]
            self._finish_inputs()
        result = native*np.asarray(self.stat['action']['std'],np.float32)+np.asarray(self.stat['action']['mean'],np.float32)
        if result.shape != (7,7) or not np.isfinite(result).all():
            raise ValueError('Malformed actual native JEPA action chunk')
        return result.astype(np.float32)
