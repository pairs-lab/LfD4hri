"""Full-bank numerical/timeline audit and stratified actual-mask visibility.

Runs on CPU while the unchanged R69 trainer owns the GPU. No data is changed.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data,phase_anchors
from VLA.flower_r69.native import action_targets,canonical_actions
from VLA.scene64_conditioning import encode_numeric
from VLA.flower_r70.common import R69,OUT,freeze_test_code,read


def array_header(archive,name):
    with archive.open(name+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
        return np.lib.format.read_array_header_2_0(stream)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    bound=freeze_test_code();data=Data(horizon=10)
    tasks=Counter();groups=defaultdict(set);targets=defaultdict(set);kinds=Counter()
    finite=[];clips=np.zeros(7,np.int64);roundtrip=0.;timeline=0.;rows=[]
    chosen,_=phase_anchors(data);selected={ei for ei,fi in chosen}
    # At least one original episode for EVERY task/object stratum, not only
    # easy reset screenshots, plus every pose corrective intervention.
    seen=set()
    for ei,r in enumerate(data.rows):
        key=(r['task'],r['pick_name'])
        if key not in seen:seen.add(key);selected.add(ei)
        if r['source_kind']=='pose_corrective':selected.add(ei)
    mask_records=[]
    for ei,r in enumerate(data.rows):
        n=data.numeric(ei);raw=n['scene64'];canonical=n['canonical7'];count=r['frames']
        if raw.shape!=(count,64) or canonical.shape!=(count,7):raise ValueError('Numeric/action timeline differs')
        enc=np.asarray([encode_numeric(v) for v in raw])
        if enc.shape!=(count,67) or not np.isfinite(canonical).all():raise ValueError('Invalid current observation/action')
        finite.append(enc);tasks[r['task']]+=count;groups[r['task']].add(r['scene_group'])
        targets[r['task']].add(r['pick_name']);kinds[r['source_kind']]+=1
        if sha256_file(r['source'])!=r['source_sha256']:raise ValueError('Source RGB/label bytes changed')
        with zipfile.ZipFile(r['source']) as z:
            headers={key:array_header(z,key) for key in ('front_rgb','wrist_rgb')}
        for shape,order,dtype in headers.values():
            if shape!=(count,384,384,3) or np.dtype(dtype)!=np.uint8:
                raise ValueError('RGB dimensions/count/channel dtype differ')
        with np.load(r['source'],allow_pickle=False) as f:
            q,closure=f['joint_positions'],f['gripper_closure']
            before=np.c_[q,closure]
            timeline=max(timeline,float(np.max(np.abs(raw[:,:7]-before))))
            if timeline>1e-6:raise ValueError('Pre-action state/encoder timeline shifted')
            if count>1:
                e=float(np.max(np.abs(q[1:]-f['next_observed_joint_positions'][:-1])))
                if e>1e-6:raise ValueError('Step-to-step measured joint continuity broken')
            if not np.allclose(canonical,f['canonical_actions'],rtol=0,atol=2e-6):
                raise ValueError('Native canonical label no longer matches source action')
            if ei in selected:
                for camera in ('front','wrist'):
                    key=camera+'_pick_mask'
                    if key not in f.files:
                        mask_records.append({'episode':ei,'task':r['task'],'camera':camera,'mask_missing':True});continue
                    masks=f[key]
                    if masks.shape!=(count,384,384):raise ValueError('Target segmentation timeline differs')
                    pixels=[]
                    for mask in masks:
                        resized=Image.fromarray(np.uint8(mask>0)*255).resize((224,224),Image.Resampling.NEAREST)
                        pixels.append(int(np.count_nonzero(np.asarray(resized))))
                    mask_records.append({'episode':ei,'task':r['task'],'camera':camera,
                        'source_kind':r['source_kind'],'target':r['pick_name'],'frames':count,
                        'visible_target_pixels_224':pixels,'zero_target_frames':sum(p==0 for p in pixels),
                        'reset_target_pixels':pixels[0],'terminal_target_pixels':pixels[-1]})
        normalized=action_targets(canonical,data.stats);decoded=canonical_actions(normalized,data.stats)
        lo,hi=(np.asarray(data.stats['adapter']['action'][k]) for k in ('q01','q99'))
        clips[:6]+=((canonical[:,:6]<lo[:6]-1e-7)|(canonical[:,:6]>hi[:6]+1e-7)).sum(0)
        expected=canonical.copy();expected[:,:6]=np.clip(expected[:,:6],lo[:6],hi[:6])
        roundtrip=max(roundtrip,float(np.max(np.abs(decoded-expected))))
        if roundtrip>1e-6:raise ValueError('FLOWER action/gripper normalization roundtrip failed')
        if ei%50==0:
            atomic_json(a.output/'progress.json',{'checked_episodes':ei+1,'planned_episodes':len(data.rows),
                'checked_frames':sum(tasks.values()),'elapsed_s':time.monotonic()-started})
    values=np.concatenate(finite);mean,std=(np.asarray(data.stats['scene67'][k]) for k in ('mean','std'))
    standardized=(values-mean)/(std+1e-8)
    visible={}
    for c in ('front','wrist'):
        rr=[r for r in mask_records if r['camera']==c and not r.get('mask_missing')]
        allp=np.asarray([p for r in rr for p in r['visible_target_pixels_224']])
        visible[c]={'episodes_with_masks':len(rr),'audited_frames':len(allp),
            'zero_target_frames':int(np.sum(allp==0)),'target_pixel_quantiles':np.quantile(allp,[0,.1,.5,.9,1]).tolist(),
            'zero_target_reset_episodes':sum(r['reset_target_pixels']==0 for r in rr)}
    result={'passed_numerical_integrity_and_timeline':True,'episodes':len(data.rows),'frames':len(values),
        'all_original_RGB_label_files_SHA_verified':True,'all_RGB_headers_count_shape_dtype_verified':True,
        'full_numeric67_causal_validation':True,'preaction_alignment_max_error':timeline,
        'action_roundtrip_max_error':roundtrip,'action_label_bounds_clip_counts':clips.tolist(),
        'frames_per_action':dict(tasks),'episodes_per_kind':dict(kinds),
        'unique_scene_groups_per_action':{k:len(v) for k,v in groups.items()},
        'unique_scene_groups_overall':len({r['scene_group'] for r in data.rows}),
        'target_names_per_action':{k:sorted(v) for k,v in targets.items()},
        'numeric_standardized_max_abs_per_column':np.max(abs(standardized),0).tolist(),
        'constant_numeric_columns':np.flatnonzero(np.std(values,0)<1e-8).tolist(),
        'mask_audit_selected_episodes':len(selected),'mask_visibility':visible,'mask_records':mask_records,
        'visibility_is_NOT_task_success_or_a_proof_of_unoccluded_all_frames':True,
        'episode_count_is_NOT_independent_scene_count':True,'CPU_only':True,
        'code':bound,'elapsed_s':time.monotonic()-started}
    atomic_json(a.output/'DATASET_AUDIT.json',result)
    atomic_json(a.output/'progress.json',{'status':'completed','checked_episodes':len(data.rows),'checked_frames':len(values)})
    print(json.dumps({k:v for k,v in result.items() if k not in ('mask_records','code','numeric_standardized_max_abs_per_column')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'dataset_reaudit')
    main(p.parse_args())
