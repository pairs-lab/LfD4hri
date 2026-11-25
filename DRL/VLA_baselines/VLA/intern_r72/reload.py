"""Actual independent reload; no in-memory or optimizer-free shortcut."""
import argparse
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.intern_r72.common import read,verify,BASE,OUT
from VLA.intern_r72.native import foundation,prediction
from VLA.intern_r72.adapters import adapt
from VLA.intern_r72.data import Data,Inputs


def load(cp):
    proof=read(cp/'TRAINING.json');verify(proof['code'])
    if sha256_file(BASE/'model.safetensors')!=read(OUT/'cpu_preflight/CPU_PREFLIGHT.json')['foundation_sha256']:
        raise ValueError('Pinned native foundation changed before independent reload')
    if not proof['completed'] or sha256_file(cp/'trainables.pt')!=proof['trainables_sha256']:
        raise ValueError('Completed same native checkpoint required')
    model,_=foundation();adapt(model,proof['adaptation']['mode'])
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    values=torch.load(cp/'trainables.pt',weights_only=True,mmap=True,map_location='cpu')
    if set(values)!=set(params):raise ValueError('Native learned key set differs')
    with torch.no_grad():
        for n,p in params.items():
            if values[n].shape!=p.shape or values[n].dtype!=torch.float32:raise ValueError('Native saved shape/master precision differs')
            p.copy_(values[n])
            if not torch.equal(p.detach().cpu(),values[n]):raise ValueError('FP32 master reload not exact')
    del values;model.requires_grad_(False)
    return model,proof,read(cp/'stats.json')


def main(a):
    torch.set_num_threads(2);torch.manual_seed(202672);torch.cuda.set_per_process_memory_fraction(.91)
    model,proof,stats=load(a.checkpoint);inputs=Inputs(stats);data=Data()
    expected=np.load(a.checkpoint/'PREDICTIONS.npy',allow_pickle=False);errors=[]
    for i,(ei,fi) in enumerate(proof['anchors']):
        got=prediction(model,inputs(data.at(ei,fi),training=False,device='cuda'))
        errors.append(float(np.abs(got-expected[i]).max()))
    result={'passed':max(errors)==0.,'fresh_process':True,'actual_predictions':len(errors),
        'prediction_max_abs_errors':errors,'all_FP32_masters_exact':True,'robot_SR_measured':False,
        'checkpoint':str(a.checkpoint.resolve()),'completed_optimizer_updates':proof['completed_optimizer_updates']}
    atomic_json(a.output,result)
    if not result['passed']:raise ValueError('Actual reloaded native predictions differ')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
