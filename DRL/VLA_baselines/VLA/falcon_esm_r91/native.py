"""Native FALCON architecture, isolated loader and explicit robot21 adaptation.

No writes to transformers, no native CLI/sudo/tokenizer fallback, no Oracle.
The full native Kosmos/ESM/action architecture is instantiated, not a small head
masquerading as the policy. GPU qualification is mandatory before using it.
"""
import importlib.util
import sys
import numpy as np
from PIL import Image
from .common import REPO, ASSETS, WEIGHT, CFG, OUT, read


def native_types():
    sys.path.insert(0, str(REPO))
    name = 'transformers.models.kosmos2.modeling_kosmos2'
    # The official wrapper copies this file over site-packages. Instead load
    # the identical author-supplied implementation into THIS process only.
    spec = importlib.util.spec_from_file_location(name, REPO/'tools/modeling_kosmos2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from falcon.model.backbone.robokosmos import RoboKosMos
    from falcon.model.policy_head.base_policy import FCDecoder_ESM
    from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth
    from falcon.model.policy_head.esm_utils.vggt.models import aggregator as esm_aggregator
    from transformers import AutoProcessor, Kosmos2Config

    def build_patch_embed_from_complete_checkpoint(self, patch_embed, img_size,
            patch_size, num_register_tokens, interpolate_antialias=True,
            interpolate_offset=0., block_chunks=0, init_values=1., embed_dim=1024):
        # Same native DINO architecture/constructor settings. The complete
        # selected checkpoint already contains all DINO weights; fetching a
        # second torch.hub foundation here is redundant and hits GitHub limits.
        if patch_embed != 'dinov2_vitl14_reg':
            raise ValueError('Only the strict selected native ESM DINO-L variant')
        self.patch_embed = esm_aggregator.vit_large(img_size=img_size,
            patch_size=patch_size, num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset, block_chunks=block_chunks,
            init_values=init_values)
        self.patch_embed.mask_token.requires_grad_(False)
    esm_aggregator.Aggregator.__build_patch_embed__ = build_patch_embed_from_complete_checkpoint

    class Native(RoboKosMos):
        def _init_backbone(self):
            self.processor = AutoProcessor.from_pretrained(ASSETS/'kosmos_assets', local_files_only=True)
            self.tokenizer = self.processor.tokenizer
            config = Kosmos2Config.from_pretrained(ASSETS/'kosmos_assets', local_files_only=True)
            return self.tokenizer, module.Kosmos2ForConditionalGeneration(config)

        def _init_heads(self):
            c = dict(self.act_head_configs)
            for key in ('type', 'esm_ckpt_path', 'use_spatial_injector'):
                c.pop(key, None)
            c.update(in_features=self.hidden_size, fwd_pred_next_n=self.fwd_pred_next_n,
                vggt_net=VGGT_Camera_Replace_Depth())
            self.latent_num = c['latent']
            return FCDecoder_ESM(**c), None, None

        def _trainable_params_setup(self):
            # Select stage trainables AFTER strict loading; never execute the
            # native llava import or rebuild a reduced action head to save RAM.
            pass

    return Native


def normalize(a, stats):
    a = np.asarray(a, np.float32).copy()
    low, high = [np.asarray(stats['actions'][k], np.float32)[:6] for k in ('q01','q99')]
    a[..., :6] = .65 * np.clip(2*(a[..., :6]-low)/(high-low)-1, -1, 1)
    a[..., 6] = (1-a[..., 6])/2  # shared signed CLOSE -> native OPEN01
    return a


def denormalize(a, stats):
    a = np.asarray(a, np.float32).copy()
    low, high = [np.asarray(stats['actions'][k], np.float32)[:6] for k in ('q01','q99')]
    a[..., :6] = (a[..., :6]/.65+1)*.5*(high-low)+low
    a[..., 6] = 1-2*(a[..., 6] > .5).astype(np.float32)
    return a


class Runtime:
    horizon = 10

    def __init__(self, stats, *, device='cuda', checkpoint=None, metadata_only=False):
        import torch
        from VLA.common.artifact_io import atomic_json
        self.stats, self.device = stats, device
        config = read(CFG)
        config['vlm']['pretrained_model_name_or_path'] = str(ASSETS/'kosmos_assets')
        config['tokenizer']['pretrained_model_name_or_path'] = str(ASSETS/'kosmos_assets')
        config['act_head'].update(camera_gt_pt=1., depth_gt_pt=1.)
        native = native_types()
        from accelerate import init_empty_weights
        # Do not globally put scalar construction on meta: native DINO builds
        # its stochastic-depth schedule with torch.linspace(...).item().
        # Empty PARAMETERS only; scalar schedules and tiny buffers remain CPU.
        with init_empty_weights(include_buffers=False):
            self.model = native(configs=config, train_setup_configs=config['train_setup'],
                act_head_configs=config['act_head'], fwd_pred_next_n=10, window_size=1,
                use_hand_rgb=True, use_state=False)
        expected = self.model.state_dict()
        raw = torch.load(WEIGHT, weights_only=True, mmap=True, map_location='cpu')['state_dict']
        state = {k.removeprefix('model.'): v for k,v in raw.items()}
        # Historical Lightning checkpoints can contain a second alias of the
        # same action head, never distinct active weights that may be discarded.
        extras = set(state)-set(expected)
        aliases = {}
        for key in extras:
            target = 'act_head.'+key[len('lm_head.'):] if key.startswith('lm_head.') else None
            if target not in state or not torch.equal(state[key], state[target]):
                raise ValueError('Unexpected nonidentical active checkpoint tensor: '+key)
            aliases[key] = target
        for key in extras: state.pop(key)
        if set(state) != set(expected):
            raise ValueError('Native checkpoint key mismatch: '+str(sorted(set(expected)-set(state))))
        for key in expected:
            if state[key].shape != expected[key].shape:
                raise ValueError('Native checkpoint shape mismatch: '+key)
        proof = {'passed': True, 'strict_native_keys':len(expected), 'verified_identical_aliases':aliases,
            'all_ESM_VLM_action_components_present':True,
            'native_ESM_parameters':sum(p.numel() for p in self.model.act_head.vggt_net.parameters()),
            'native_unique_parameters':sum(p.numel() for p in self.model.parameters())}
        if metadata_only:
            atomic_json(OUT/'STRICT_META_LOAD.json', proof)
            return
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.to(device=device, dtype=torch.bfloat16 if device=='cuda' else torch.float32)
        del state, raw, expected
        self.model.requires_grad_(False)
        self.model.robot21 = torch.nn.Sequential(torch.nn.Linear(21,1024), torch.nn.SiLU(),
            torch.nn.Linear(1024,self.model.hidden_size)).to(device=device, dtype=torch.float32)
        # Stable initialization of extra causal input, not an action residual.
        torch.nn.init.zeros_(self.model.robot21[-1].weight)
        torch.nn.init.zeros_(self.model.robot21[-1].bias)
        self._state = None
        def causal_condition(module, args):
            if self._state is None: raise ValueError('Missing current robot21')
            feature = self.model.robot21(self._state)
            return (args[0]+feature.to(args[0].dtype)[:,None,None,:], *args[1:])
        self.model.act_head.register_forward_pre_hook(causal_condition)
        self.model.backbone.config.use_cache = False
        self.last_input = {}
        if checkpoint is not None:
            values = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
            targets = dict(self.model.named_parameters())
            for name, value in values.items():
                if name not in targets or targets[name].shape != value.shape:
                    raise ValueError('Adaptation shape/key mismatch: '+name)
                targets[name].data.copy_(value.to(device=targets[name].device, dtype=targets[name].dtype))
            self.loaded_adaptation_keys = set(values)
        self.load_proof = proof

    def stage(self, name):
        import torch
        self.model.requires_grad_(False)
        self.model.robot21.requires_grad_(True)
        self.model.act_head.spatial_injector.requires_grad_(True)
        if name in ('native_embodiment', 'joint_refinement'):
            # Target-domain native output calibration precedes spatial-only
            # refinement. This is an explicitly declared embodiment adaptation.
            for key in ('mlp','actions','gripper'):
                getattr(self.model.act_head,key).requires_grad_(True)
        if name == 'joint_refinement':
            from peft import LoraConfig, inject_adapter_in_model
            if not hasattr(self,'lora_installed'):
                config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                    target_modules=['q_proj','k_proj','v_proj','out_proj','fc1','fc2'])
                inject_adapter_in_model(config, self.model.backbone)
                self.lora_installed = True
            for key,p in self.model.backbone.named_parameters():
                if 'lora_' in key: p.requires_grad_(True)
            self.model.backbone.gradient_checkpointing_enable()
            self.model.backbone.enable_input_require_grads()
        # Trainable masters always FP32; frozen backbone/ESM remains BF16.
        for p in self.model.parameters():
            if p.requires_grad: p.data = p.data.float()
        self.model.act_head.vggt_net.requires_grad_(False)
        return {k:p for k,p in self.model.named_parameters() if p.requires_grad}

    def batch(self, s):
        import torch
        from torchvision.transforms import functional as F, InterpolationMode
        tokens = self.model.tokenizer([f'<grounding>An image of a robot {s["instruction"].strip()}'],
            padding='longest', truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1]>512: raise ValueError('Do not truncate requested instruction')
        def rgb(image):
            image = Image.fromarray(image).convert('RGB')
            image = F.resize(image, [224,224], interpolation=InterpolationMode.BICUBIC)
            return F.normalize(F.to_tensor(image), [.48145466,.4578275,.40821073],
                [.26862954,.26130258,.27577711])[None,None].to(self.device)
        # Exact native geometrical crop/resize, including adjusted K.
        from falcon.data.data_utils import process_sequence_data
        image, depth, K, mask = process_sequence_data(s['front'][None],
            s['depth_z_m'][None], s['intrinsics'][None], target_resolution=(224,224))
        # Author's helper returns RGB in0..255, whereas the ESM API and mean/std
        # require0..1. Validate/declare this conversion instead of feeding255.
        if image.max()>1: image = image/255.
        mask = torch.isfinite(depth[...,0]) & (depth[...,0]>.01) & (depth[...,0]<3.)
        depth = torch.where(mask[...,None],depth,torch.zeros_like(depth))
        from protocol21.state21_reference import normalize
        self._state = torch.as_tensor(normalize(s['raw_state21'],self.stats['robot21'])[None],
            device=self.device,dtype=torch.float32)
        self.last_input = {'actual_RGB_views':2,'native_ESM':True,'actual_depth':True,
            'robot21':True,'instruction_tokens':int(tokens['input_ids'].shape[1]),
            'teacher_phase_future_labels_or_geometric_actions':False}
        return dict(vision_x=rgb(s['front']), vision_gripper=rgb(s['wrist']),
            lang_x=tokens['input_ids'].to(self.device), attention_mask=tokens['attention_mask'].bool().to(self.device),
            rgb_vggt=image[None].to(self.device), static_depth_vggt=depth[None].to(self.device),
            point_masks_vggt=mask[None].to(self.device),
            static_cam_intr_mat_vggt=K[None].to(self.device),
            static_cam_ex_mat_vggt=torch.as_tensor(s['extrinsics_OpenCV'][None,None,:3],device=self.device))

    def forward(self,s):
        import torch
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda'):
            return self.model.forward_continuous(**self.batch(s), mode='inference')

    def loss(self,s):
        import torch
        self.model.train();self.model.act_head.vggt_net.eval()
        prediction = self.forward(s)
        target = normalize(s['action'],self.stats)
        labels = (torch.as_tensor(target[None,None,:,:6],device=self.device),
            torch.as_tensor(target[None,None,:,6],device=self.device))
        mask = torch.as_tensor((~s['action_is_pad'])[None,None],device=self.device)
        losses = self.model.act_head.loss(tuple(x.float() for x in prediction),labels,mask)
        return losses['loss_arm']+.01*losses['loss_gripper'],losses

    def predict(self,s):
        import torch
        self.model.eval()
        with torch.no_grad():
            arm, grip = self.forward(s)
            values = torch.cat((arm.float(),grip.float().sigmoid()),-1).cpu().numpy().reshape(10,7)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite native FALCON action')
        return denormalize(values,self.stats)
