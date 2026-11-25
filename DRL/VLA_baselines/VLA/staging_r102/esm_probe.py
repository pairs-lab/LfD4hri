"""Authentic frozen ESM component inference; NOT a full policy or robot SR."""
import gc,os,time
from types import SimpleNamespace
import numpy as np
from .common import OUT,R100,read,utc,atomic_json,geometry_contract
from VLA.common.provenance import sha256_file


def main():
    os.nice(10)
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoProcessor
    from VLA.falcon_esm_r91.common import ASSETS,WEIGHT,WEIGHT_SHA
    from VLA.falcon_esm_r91.native import native_types,Runtime as NativeRuntime
    from VLA.admission_r100.falcon_native import Data,align_esm_patch_tokens
    torch.set_num_threads(1)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(4096*2**20/total)
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    atomic_json(OUT/'ESM_STATUS.json',{'UTC':utc(),'status':'loading_authentic_ESM_component',
        'actual_neural_policy_episodes':0,'CUDA_allocator_cap_MiB':4096,'pid':os.getpid()})
    # Use the same authentic DINO constructor selected by the qualified full model.
    native_types()
    from falcon.model.policy_head.esm_utils.vggt.models.vggt_camera_replace_depth import VGGT_Camera_Replace_Depth
    if sha256_file(WEIGHT)!=WEIGHT_SHA:raise ValueError('Authentic foundation bytes changed')
    with init_empty_weights(include_buffers=False):esm=VGGT_Camera_Replace_Depth()
    raw=torch.load(WEIGHT,weights_only=True,mmap=True,map_location='cpu')['state_dict']
    prefix='model.act_head.vggt_net.'
    weights={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
    if set(weights)!=set(esm.state_dict()):raise ValueError('Native ESM component key mismatch')
    esm.load_state_dict(weights,strict=True,assign=True)
    esm.to(device='cuda',dtype=torch.bfloat16);esm.requires_grad_(False);esm.eval()
    del weights,raw;gc.collect()
    aggregator=esm.aggregator
    aggregator.patch_embed.register_forward_hook(lambda module,args,value:
        align_esm_patch_tokens(value,aggregator.depth_placeholder.dtype))
    data=Data();stats=read(R100/'falcon_esm/DATA_STATS.json')
    # Exercise the actual production batch function, not another image/depth recipe.
    processor=AutoProcessor.from_pretrained(ASSETS/'kosmos_assets',local_files_only=True)
    runtime=NativeRuntime.__new__(NativeRuntime);runtime.device='cuda';runtime.stats=stats
    runtime.model=SimpleNamespace(tokenizer=processor.tokenizer);runtime.last_input={}
    keys=set();anchors=[]
    for ei,row in enumerate(data.rows):
        key=row['task'],row['pick_name']
        if row['source_kind']!='complete_original' or key in keys:continue
        keys.add(key)
        anchors.extend((ei,fi) for fi in sorted({0,row['frames']//2,row['frames']-1}))
    if len(keys)!=76 or len(anchors)!=228:raise ValueError('Actual four-action/19-target coverage missing')
    records=[];torch.cuda.reset_peak_memory_stats()
    for ei,fi in anchors:
        s=data.at(ei,fi);geometry_contract(s['intrinsics'],s['extrinsics_OpenCV'])
        batch=runtime.batch(s)
        kwargs={'images':batch['rgb_vggt'],'extrinsics':batch['static_cam_ex_mat_vggt'],
            'intrinsics':batch['static_cam_intr_mat_vggt'],'depth':batch['static_depth_vggt'],
            'mask':batch['point_masks_vggt'],'camera_gt_pt':1.,'depth_gt_pt':1.}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictions,tokens=esm.inference(**kwargs)
        spatial=tokens[-1][:,:,5:,:]
        if spatial.shape!=(1,1,256,2048) or not torch.isfinite(spatial).all():
            raise ValueError('Malformed/nonfinite actual native spatial tokens')
        if float(spatial.float().std())<=0:raise ValueError('Constant native ESM features')
        if not all(torch.isfinite(v).all() for v in predictions.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Nonfinite actual native ESM auxiliary output')
        records.append({'episode':ei,'frame':fi,'task':s['task'],'target':data.rows[ei]['pick_name'],
            'spatial_shape':list(spatial.shape),'spatial_std':float(spatial.float().std()),
            'depth_valid_pixels':int(batch['point_masks_vggt'].sum()),
            'native_67_finite':bool(torch.isfinite(runtime._state).all())})
        del spatial,tokens,predictions,batch,kwargs
        if len(records)%4==0:
            atomic_json(OUT/'ESM_STATUS.json',{'UTC':utc(),'status':'running_actual_ESM_component',
                'completed_cases':len(records),'planned_cases':len(anchors),
                'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,
                'actual_neural_policy_episodes':0,'elapsed_s':time.monotonic()-started})
    proof={'passed':True,'UTC':utc(),'cases':records,'task_target_strata':len(keys),
        'authentic_ESM_parameters':sum(p.numel() for p in esm.parameters()),
        'peak_GPU_MiB':torch.cuda.max_memory_allocated()/2**20,'foundation_sha256':WEIGHT_SHA,
        'real_native_RGBD_67_processor_and_ESM_inference':True,
        'full_policy_or_new_action_SR_NOT_tested_by_this_component_probe':True,
        'actual_neural_policy_episodes':0,'elapsed_s':time.monotonic()-started}
    atomic_json(OUT/'ESM_COMPONENT_PROOF.json',proof)
    atomic_json(OUT/'ESM_STATUS.json',{'UTC':utc(),'status':'completed','cases':len(records),
        'peak_GPU_MiB':proof['peak_GPU_MiB'],'actual_neural_policy_episodes':0})


if __name__=='__main__':main()
