"""Native policy factories. No fixture/random controller is an inference fallback.

Resource root is an external DRL-like directory with VLA/third_party and
VLA/checkpoints. It is read only. Each family should use its own process/env.
"""
import importlib
from pathlib import Path
import sys
import numpy as np

from .specs import SPECS


PACKAGE = Path(__file__).resolve().parents[1]


def configure(family, root):
    root = Path(root).resolve()
    repos = {'flower': 'flower_vla_pret', 'xvla': 'lerobot', 'smolvla': 'lerobot',
             'vla_adapter': 'vla_adapter', 'pi0': 'openpi', 'vla_jepa': 'lerobot_jepa',
             'dita': 'dita', 'vlanext_s': 'vlanext', 'falcon_esm': 'falcon_vla', 'seer': 'seer'}
    repo = root/'VLA/third_party'/repos[family]
    if not repo.is_dir():
        raise FileNotFoundError('Restore native source before inference: ' + str(repo))
    for directory in (repo, repo/'src'):
        if directory.is_dir():
            sys.path.insert(0, str(directory))
    # Configure resource paths BEFORE importing modules that snapshot constants.
    if family in ('dita', 'vlanext_s', 'falcon_esm', 'seer', 'vla_jepa'):
        names = {'dita': 'dita_transfer_r84', 'vlanext_s': 'replacement_r84',
                 'falcon_esm': 'falcon_esm_r91', 'seer': 'admission_r100', 'vla_jepa': 'extension_r65'}
        common = importlib.import_module('VLA.'+names[family]+'.common')
        common.REPO = repo
        if family == 'dita':
            common.BASE = root/'VLA/checkpoints/replacement_r84/dita/official_oxe_policy.pth'
            common.TEXT = root/'VLA/checkpoints/replacement_r84/dita_clip_text'
            common.OUT = root/'VLA/audits/replacement_r84_20261020/dita'
        elif family == 'vlanext_s':
            common.BASE = root/'VLA/checkpoints/replacement_r84/qwen3.5-0.8b'
        elif family == 'falcon_esm':
            common.ASSETS = root/'VLA/checkpoints/falcon_esm_r91'
            common.WEIGHT = common.ASSETS/'falcon_esm_fc_calvin_abc_wdepth.pt'
            common.CFG = repo/'configs/calvin_finetune/finetune_kosmos_cont-fc-post_esm_wd-0_use-hand_ws-1_act-10.json'
        elif family == 'seer':
            common.ASSETS = root/'VLA/checkpoints/admission_r100'
            common.OUT = root/'VLA/audits/native_admission_r100_20261025'
        elif family == 'vla_jepa':
            common.BASE = root/'VLA/checkpoints/vla_jepa_pretrain_r65'
            common.QWEN = root/'VLA/checkpoints/vla_jepa_qwen_metadata_r65'
    return root, repo


def create(family, stats, *, resources=PACKAGE, foundation=None, device='cuda'):
    """Load authentic released weights then adapt state conditioning; no optimizer step."""
    if family not in SPECS:
        raise ValueError(family)
    from .resources import check
    missing=check(family,resources,foundation)['missing']
    if missing:
        raise FileNotFoundError('Missing local native resources: '+str(missing))
    if device != 'cuda' and family not in ('flower', 'dita', 'falcon_esm', 'seer'):
        raise ValueError('This native family currently requires its CUDA environment')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU/mock-policy substitution will be made')
    root, repo = configure(family, resources)
    if family == 'flower':
        # These globals are referenced only at construction, not import time.
        sys.path.insert(0, str(repo))
        m = importlib.import_module('VLA.flower_r69.native')
        m.SOURCE = repo; m.BASE = Path(foundation) if foundation else root/'VLA/checkpoints/candidates_r68/flower'
        model, tokenizer, receipt = m.load_foundation()
        m.adapt(model, device=device)
        if device == 'cpu':
            # CPU smoke uses FP32 throughout; CUDA keeps native BF16 compute.
            model.float()
        class Flower:
            def __init__(self):
                self.model, self.stats = model, stats
            def loss(self, s):
                import torch
                batch = m.native_batch(s, tokenizer, stats, device)
                batch['actions'] = torch.as_tensor(m.action_targets(s['action'], stats)[None], device=device)
                from VLA.flower_r69.loss import loss as masked_flow_loss
                model.train(); model.vlm.eval()
                pad = torch.as_tensor(s['action_is_pad'][None], device=device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    return masked_flow_loss(model, model.encode_observations(batch), batch['actions'], pad)
            def predict(self, s, seed=0):
                import torch
                with torch.random.fork_rng(devices=[0] if device == 'cuda' else []), torch.inference_mode(), \
                     torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    torch.manual_seed(seed); model.eval()
                    cond = model.encode_observations(m.native_batch(s, tokenizer, stats, device))
                    noise = torch.randn((1,10,7), device=device)
                    result = model.sample_actions(noise, cond, inference=True)
                    return m.canonical_actions(result[0].float().cpu().numpy(), stats)
        return Flower()
    if family in ('xvla', 'vla_adapter'):
        m = importlib.import_module('VLA.new_vlas.runtime')
        cls = m.XVLARuntime if family == 'xvla' else m.AdapterRuntime
        obj = cls(stats, checkpoint=foundation, resource_root=root)
        return obj.to(device)
    if family == 'smolvla':
        if foundation is None:
            raise ValueError('SmolVLA --foundation must point to the resolved local smolvla_base snapshot')
        return Smol(stats, Path(foundation), device)
    if family == 'pi0':
        if foundation is None:
            raise ValueError('pi0 --foundation must point to an actual adapted OpenPI checkpoint (params/ + assets/)')
        from .pi0 import Runtime
        return Runtime(stats, Path(foundation))
    if family == 'vla_jepa':
        m = importlib.import_module('VLA.extension_r65.jepa')
        if foundation: m.BASE = Path(foundation)
        return m.Runtime({'state': stats['robot21'], 'action': stats['action']})
    if family == 'dita':
        m = importlib.import_module('VLA.dita_transfer_r84.native')
        if foundation: m.BASE = Path(foundation)
        return m.Runtime(stats, device=device)
    if family == 'vlanext_s':
        m = importlib.import_module('VLA.replacement_r84.native')
        if foundation: m.BASE = Path(foundation)
        return m.Runtime(stats)
    if family == 'falcon_esm':
        m = importlib.import_module('VLA.falcon_esm_r91.native')
        if foundation: m.WEIGHT = Path(foundation)
        obj = m.Runtime(stats, device=device)
        obj.stage('joint_refinement')
        # Preserve native ESM dtype boundary fix from admission implementation.
        from VLA.admission_r100.falcon_native import align_esm_patch_tokens
        aggregator = obj.model.act_head.vggt_net.aggregator
        aggregator.patch_embed.register_forward_hook(lambda module, args, out:
            align_esm_patch_tokens(out, aggregator.depth_placeholder.dtype))
        if device == 'cuda':
            obj.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return obj
    m = importlib.import_module('VLA.admission_r100.seer_native')
    m.REPO = repo
    return m.Runtime(stats, device=device, foundation=foundation)


class Smol:
    def __init__(self, stats, foundation, device):
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
        from VLA.smolvla.precision import fp32_policy_class
        from VLA.family_transfer_r54.smol import processors
        original_config = SmolVLAConfig.from_pretrained(str(foundation))
        original_config.train_expert_only = False
        original_config.freeze_vision_encoder = True
        original_config.train_state_proj = True
        original_config.compile_model = False
        self.model = fp32_policy_class(True).from_pretrained(str(foundation),config=original_config,
                                                          strict=True,local_files_only=True)
        cfg = self.model.config
        old = self.model.model.state_proj
        new = torch.nn.Linear(21, old.out_features, bias=old.bias is not None)
        torch.nn.init.zeros_(new.weight)
        if old.bias is not None:
            new.bias.data.copy_(old.bias.detach().float())
        self.model.model.state_proj = new
        cfg.max_state_dim = 21
        cfg.chunk_size = cfg.n_action_steps = 5
        cfg.device = device; cfg.empty_cameras = 0
        cfg.adapt_to_pi_aloha = False; cfg.use_delta_joint_actions_aloha = False
        cfg.num_steps = 10; cfg.freeze_vision_encoder = True
        cfg.input_features = {f'observation.images.camera{i}': PolicyFeature(type=FeatureType.VISUAL, shape=(3,384,384)) for i in (1,2)}
        cfg.input_features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(21,))
        cfg.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        cfg.resize_imgs_with_padding = (512,512)
        self.model.to(device); self.device = device
        self.pre, self.post = processors(cfg, {'observation.state': stats['robot21'], 'action': stats['action']})

    def batch(self, s, training=False):
        import torch
        b = {f'observation.images.camera{i}': torch.from_numpy(s[k].copy()).permute(2,0,1)[None].float()/255
             for i,k in ((1,'front'), (2,'wrist'))}
        b['observation.state'] = torch.as_tensor(s['raw_state21'])[None]
        b['task'] = [s['instruction']]
        if training:
            b['action'] = torch.as_tensor(s['action'])[None]
            b['action_is_pad'] = torch.as_tensor(s['action_is_pad'])[None]
        b = self.pre(b)
        return {k:v.to(self.device) if hasattr(v,'to') else v for k,v in b.items()}

    def loss(self, s):
        import torch
        self.model.train()
        with torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            return self.model(self.batch(s, training=True))[0]

    def predict(self, s, seed=0):
        import torch
        self.model.eval(); self.model.reset()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1,5,self.model.config.max_action_dim), generator=generator, device=self.device)
        with torch.inference_mode(),torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            return self.post(self.model.predict_action_chunk(self.batch(s),noise=noise))[0].float().cpu().numpy()
