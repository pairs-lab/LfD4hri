"""Independent process reload of a completed FLOWER calibration/fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data
from VLA.flower_r69.native import load_foundation, adapt, native_batch
from VLA.flower_r69.gate import prediction


def main(a):
    cp=a.checkpoint.resolve();proof=json.loads((cp/'TRAINING.json').read_text())
    if not proof['completed']:raise ValueError('Completed actual fit required')
    for p,digest in proof['code'].items():
        if sha256_file(p)!=digest:raise ValueError('Saved fit source changed: '+p)
    if sha256_file(cp/'trainables.pt')!=proof['trainables_sha256']:
        raise ValueError('Saved learned tensor file changed')
    torch.set_num_threads(2);torch.manual_seed(202669)
    torch.cuda.set_per_process_memory_fraction(.79)
    model,tokenizer,_=load_foundation();adapt(model)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    saved=torch.load(cp/'trainables.pt',weights_only=True,mmap=True,map_location='cpu')
    if set(saved)!=set(params):raise ValueError('Full trainable key set differs')
    with torch.no_grad():
        for n,p in params.items():
            value=saved[n]
            if value.shape!=p.shape or value.dtype!=torch.float32:raise ValueError('Master shape/precision differs')
            p.copy_(value)
            if not torch.equal(p.detach().cpu(),value):raise ValueError('Learned master is not exact')
    data=Data(horizon=10);expected=np.load(cp/'PREDICTIONS.npy',allow_pickle=False);errors=[]
    for index,(ei,fi) in enumerate(proof['anchors']):
        actual=prediction(model,native_batch(data.at(ei,fi),tokenizer,data.stats))
        errors.append(float(np.max(np.abs(actual-expected[index]))))
    result={'passed':max(errors)==0.,'fresh_process':True,'checkpoint':str(cp),
        'actual_new_optimizer_updates':proof['actual_new_optimizer_updates'],
        'prediction_max_abs_errors':errors,'saved_FP32_masters_exact':True,
        'robot_success_rate_measured':False}
    atomic_json(a.output,result)
    if not result['passed']:raise ValueError('Native fresh-process prediction not exact')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
