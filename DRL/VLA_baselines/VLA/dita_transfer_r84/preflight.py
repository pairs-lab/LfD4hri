"""CPU native checkpoint/chronology/action/recipe checks before any expensive fit."""
import numpy as np
from .common import OUT,ROOT,REPO,TEXT,read,bindings,verify
from .native import Runtime
from .data import sample
from VLA.common.artifact_io import atomic_json
from VLA.replacement_r84.data import Data

def main():
    import torch
    torch.set_num_threads(2)
    stats=read(ROOT/'VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json')['stats']
    data=Data()
    from VLA.family_transfer_r54.data import phase_anchors
    ids,_=phase_anchors(data)
    tested=[]
    for ei,fi in ids:
        s=sample(data,ei,fi,stats)
        n=data.numeric(ei);raw_ids=s['action_label_indices']
        valid=s['action_valid11']
        np.testing.assert_array_equal(s['history_indices'],np.maximum([fi-1,fi],0))
        np.testing.assert_allclose(s['normalized_action11'][valid,6],
            (1-n['canonical7'][raw_ids[valid],6])/2,atol=1e-7)
        if not (s['history74'].shape==(2,74) and valid[1] and raw_ids[1]==fi):
            raise ValueError('Native current/future alignment or measured history incomplete')
        for name,images in s['history_images'].items():
            if len(images)!=2 or any(im.shape!=(384,384,3) or im.dtype!=np.uint8 for im in images):
                raise ValueError('Missing physical image history '+name)
        tested.append([ei,fi])
    model=Runtime(stats,device='cpu',load_text=False)
    state=model.model['native'].state_dict()
    native=torch.load(__import__('VLA.dita_transfer_r84.common',fromlist=['BASE']).BASE,
        weights_only=True,map_location='cpu',mmap=True)['parameter']
    for key,v in native.items():
        if not torch.equal(state[key],v):raise ValueError('Native pretrained tensor changed: '+key)
    scheduler=model.model['native'].noise_scheduler.config
    atomic_json(OUT/'CPU_PREFLIGHT.json',{'passed':True,'source_revision':read(OUT/'ARTIFACT_ADMISSION.json')['source_revision'],
        'native_strict_state_keys':len(state),'native_checkpoint_values_exact':True,
        'native_parameters':sum(p.numel() for p in model.model['native'].parameters()),
        'new_sensor74_parameters':sum(p.numel() for p in model.model['sensor74'].parameters()),
        'native_backbone_or_denoiser_replaced':False,'native_gripper':'absolute OPEN01; >0.5 open',
        'history_images_and_numeric':'two real previous/current observations, reset repeat',
        'horizon':10,'native_diffusion_slots_including_predicted_past_slot':11,
        'native_scheduler_config':dict(scheduler),'phase_samples_checked':tested,
        'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),
        'native_visual_normalization':'DINOv2 ImageNet mean/std, not FLOWER CLIP preprocessing',
        'explicit_extensions':['shared native image/QFormer encoder on front/wrist/privileged map',
            'zero-initialized sensor74 token alongside native in-context visual tokens',
            'TRAIN domain pose normalization and true-tail loss mask',
            'native explicit attention fallback enabled; dead unused CLIP vision import removed'],
        'original_recipe':'official full-FT denoiser/DINOv2/QFormer, epsilon DDPM100, DDIM10',
        'resource_adjusted_budget_not_paper_reproduction':True,'actual_GPU_gate_or_robot_SR':False,
        'code':bindings()})

if __name__=='__main__':main()
