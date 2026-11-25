"""Read every numeric TRAIN frame; show actual policy RGB, never fake images."""
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw

from .common import OUT,PARENT,TASKS,read,atomic_json
from .data import Data,regimes
from VLA.replacement_r84.data import sample,observation74,normalize_actions,denormalize_actions
from VLA.new_vlas.action_contracts import native8
from VLA.common.provenance import sha256_file


def main():
    OUT.mkdir(exist_ok=True)
    data=Data();stats=read(PARENT/'TRAINING.json')['stats']
    totals=Counter();per=defaultdict(Counter);coverage=defaultdict(set)
    selected={};names=('safe_lateral_approach','contact_or_grip_transition','held_transport')
    checked=0;prefix_error=0.;action_error=0.
    for ei,row in enumerate(data.rows):
        n=data.numeric(ei);raw,a=n['scene64'],n['canonical7']
        if len(raw)!=row['frames'] or not np.isfinite(raw).all() or not np.isfinite(a).all():
            raise ValueError('Malformed or nonfinite actual data')
        if set(np.unique(a[:,6]))-{-1.,1.}:raise ValueError('Canonical gripper is not signed close')
        # Native cache conversion agrees with the live server representation.
        canonical_columns=n['state20'][:,3:9][:,[0,2,4,1,3,5]]
        live=native8(n['state20'][:,:3],canonical_columns,raw[:,27])
        err=float(np.max(abs(live-n['native8'])))
        prefix_error=max(prefix_error,err)
        if err>1e-6:raise ValueError('Native8 TRAIN/live state representation mismatch')
        decoded=denormalize_actions(normalize_actions(a,stats),stats)
        # Quantile clipping is expected and measured, not treated as an inverse everywhere.
        lo,hi=(np.asarray(stats['actions'][k]) for k in ('q01','q99'))
        wanted=a.copy();wanted[:,:6]=np.clip(a[:,:6],lo[:6],hi[:6])
        action_error=max(action_error,float(np.max(abs(decoded-wanted))))
        for name,mask in regimes(raw,a).items():
            count=int(mask.sum());totals[name]+=count;per[row['task']][name]+=count
            if count:
                coverage[row['task']+'/'+name].add(row['pick_name'])
                key=row['task'],name
                if key not in selected and row['source_kind']=='complete_original':
                    selected[key]=(ei,int(np.flatnonzero(mask)[0]))
        for fi in sorted({0,len(raw)//2,len(raw)-1}):
            expected=np.maximum(np.arange(fi-7,fi+1),0)
            history=np.stack([observation74(n['native8'][t,:6],raw[t],stats) for t in expected])
            if history.shape!=(8,74) or not np.isfinite(history).all():raise ValueError('History invalid')
            if np.max(expected)>fi or np.min(expected)<0:raise ValueError('Future/history leakage')
            checked+=1
    if prefix_error>1e-6 or action_error>1e-6:raise ValueError('Representation parity failed')
    if {r['task'] for r in data.rows}!=set(TASKS):raise ValueError('Four tasks required')
    for task in TASKS:
        if len(data.groups[task]['complete_original'])!=19:raise ValueError('Need19 targets per task')
        if task!='touch' and not per[task]['held_transport']:raise ValueError('Held transport labels absent')
    entries=[(task,mode,*selected[task,mode]) for task in TASKS for mode in names if (task,mode) in selected]
    canvas=Image.new('RGB',(768,280*len(entries)),'white');draw=ImageDraw.Draw(canvas);records=[]
    for ri,(task,mode,ei,fi) in enumerate(entries):
        s=sample(data,ei,fi,stats);row=data.rows[ei]
        if not np.array_equal(s['history74'][-1],observation74(data.numeric(ei)['native8'][fi,:6],s['raw64'],stats)):
            raise ValueError('Actual sampled current frame is not history endpoint')
        draw.text((3,ri*280+3),f'{task} {mode}: {row["pick_name"]} t{fi}',fill='black')
        records.append({'task':task,'regime':mode,'episode':ei,'frame':fi,
            'source':row['source'],'source_sha256':sha256_file(row['source']),
            'instruction':s['instruction'],'model_history_shape':list(s['history74'].shape)})
        for col,key in enumerate(('front','wrist','map')):
            im=np.asarray(s[key])
            if im.ndim!=3 or im.shape[-1]!=3 or im.dtype!=np.uint8:raise ValueError('RGB schema mismatch')
            canvas.paste(Image.fromarray(im).resize((256,256)),(col*256,ri*280+24))
    canvas.save(OUT/'TRAIN_MODEL_INPUT_CONTACT_SHEET.png')
    rng=np.random.default_rng(202699);draws=Counter();sources=Counter()
    for _ in range(12000):
        ei,fi=data.draw(rng);row=data.rows[ei]
        if not 0<=fi<row['frames']:raise ValueError('Invalid sampler window')
        draws[row['task']+'/'+row['pick_name']]+=1;sources[row['source_kind']]+=1
    if len(draws)!=76 or any(v<70 for v in draws.values()):raise ValueError('Sampler lost task/target support')
    atomic_json(OUT/'TRAIN_DATA_AUDIT.json',{'passed':True,'episodes':len(data.rows),
        'frames':sum(r['frames'] for r in data.rows),'history_cases_checked':checked,
        'native8_live_conversion_max_error':prefix_error,'action_decode_clipped_inverse_error':action_error,
        'regime_frames':dict(totals),'per_action':{k:dict(v) for k,v in per.items()},
        'target_coverage':{k:len(v) for k,v in coverage.items()},'actual_RGB_examples':records,
        'sampler_actual_draws':12000,'sampled_target_counts':dict(draws),
        'sampled_source_counts':dict(sources),'sampled_regime_counts':dict(data.stage_exposure),
        'training_data_only_no_R98_rollout_labels_added':True,
        'images_are_actual_policy_RGB_display_resized_only':True})
    print({'passed':True,'episodes':len(data.rows),'frames':sum(r['frames'] for r in data.rows),
        'regimes':dict(totals),'contact_sheet':str(OUT/'TRAIN_MODEL_INPUT_CONTACT_SHEET.png')},flush=True)


if __name__=='__main__':main()
