"""All-episode language/FAST budget audit without decoding RGB repeatedly."""
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.intern_r72.common import OUT,read
from VLA.intern_r72.data import Data,Inputs,targets,measured_feedback
from VLA.scene64_conditioning import encode_numeric
from lerobot.transforms.constants import SYSTEM_MESSAGE


def main():
    out=OUT/'TOKEN_BUDGET_AUDIT.json'
    if out.exists():raise FileExistsError('Keep previous token audit')
    data=Data();stats={**data.stats,'intern_action':data.intern_action_stats};inputs=Inputs(stats)
    mean,std=(np.asarray(stats['scene67'][k],np.float32) for k in ('mean','std'))
    receipts=read(OUT/'cpu_preflight/CPU_PREFLIGHT.json')['receipts']
    reference={(r['episode'],r['frame']):r for r in receipts}
    offset=sum((t*h*w//4-1) for t,h,w in receipts[0]['image_grid_thw'])
    checked=[];verified_reference=0;worst=None;max_roundtrip=0.
    for ei,row in enumerate(data.rows):
        n=data.numeric(ei);raw,cmd=n['scene64'],n['canonical7']
        frames={0,len(raw)-1}
        for flag in (raw[:,28:30].any(axis=1), (cmd[:,6]>.2)&(raw[:,10]<.5)):
            ids=np.flatnonzero(flag)
            if len(ids):frames.add(int(ids[0]))
        frames.update(fi for epi,fi in reference if epi==ei)
        for fi in sorted(frames):
            state=raw[fi];continuous=(encode_numeric(state)-mean)/(std+1e-8)
            bins=np.digitize(continuous/3.,np.linspace(-1,1,257)[:-1])-1
            text=('Task: '+row['instruction']+'; '+measured_feedback(state)+
                '; Control Mode: <end_effector>; State: '+' '.join(map(str,bins))+'; Output: <Action>')
            messages=[{'role':'system','content':SYSTEM_MESSAGE},{'role':'user','content':[
                *[{'type':'image'} for _ in range(3)],{'type':'text','text':text}]}]
            prompt=inputs.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            length=len(inputs.processor.tokenizer(prompt)['input_ids'])+offset
            ids=fi+np.arange(50);pad=ids>=len(raw)
            action=targets(cmd[np.minimum(ids,len(raw)-1)],pad,stats['intern_action'])
            horizon=int((~pad).sum());fast=inputs.fast(action[None,:horizon])[0]
            # Each robot_action token is one token, EOS is exactly one token.
            total=length+len(fast)+1
            if (ei,fi) in reference:
                r=reference[ei,fi]
                if length!=r['prompt_length'] or total!=r['language_length']:
                    raise ValueError('Language-only token reservation differs from actual Qwen image processor')
                verified_reference+=1
            value={'episode':ei,'frame':fi,'task':row['task'],'kind':row['source_kind'],
                'prompt_length':length,'FAST_tokens':len(fast),'training_length':total,'real_horizon':horizon}
            checked.append(value)
            if worst is None or total>worst['training_length']:worst=value
            if len(checked)%100==0:
                reconstructed=inputs.fast.decode([fast],time_horizon=horizon,action_dim=32)[0]
                max_roundtrip=max(max_roundtrip,float(np.abs(reconstructed[:,:7]-action[:horizon,:7]).max()))
        if (ei+1)%200==0:
            atomic_json(OUT/'TOKEN_BUDGET_PROGRESS.json',{'episodes':ei+1,'sampled_windows':len(checked),'worst':worst})
    result={'passed':worst['training_length']<=1024,'episodes_covered':len(data.rows),
        'windows_checked':len(checked),'scope':'every TRAIN episode reset/terminal/first-close/first-contact, NOT every119495 sliding window',
        'actual_processor_reference_cases_verified':verified_reference,'worst':worst,
        'old650_budget_overflow_windows':sum(r['training_length']>650 for r in checked),
        'explicit1024_budget_overflow_windows':sum(r['training_length']>1024 for r in checked),
        'FAST_quantization_max_abs_normalized_action_error_sample':max_roundtrip,
        'all_windows_keep_runtime_no_silent_truncation_guard':True,'windows':checked}
    atomic_json(out,result)
    if not result['passed']:raise ValueError('Token budget insufficient; do not start training')
    print({k:v for k,v in result.items() if k!='windows'},flush=True)


if __name__=='__main__':main()
