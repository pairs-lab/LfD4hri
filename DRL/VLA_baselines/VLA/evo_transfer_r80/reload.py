"""Fresh interpreter, exact learned weights plus deterministic neural predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import read, verify
from .native import Runtime
from .data import Data


def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.82)
    proof=read(a.checkpoint/'TRAINING.json');verify(proof['code'])
    if not proof['completed'] or sha256_file(a.checkpoint/'learned.safetensors')!=proof['learned_sha256']:
        raise ValueError('Incomplete or changed native learned checkpoint')
    runtime=Runtime(read(a.checkpoint/'stats.json'),a.checkpoint);data=Data()
    anchors=read(a.checkpoint/'PREDICTION_ANCHORS.json')
    expected=np.load(a.checkpoint/'PREDICTIONS.npy',allow_pickle=False)
    errors=[]
    for i,(ei,fi) in enumerate(anchors):
        actual=runtime.predict(data.at(ei,fi),202680+i)
        errors.append(float(np.max(np.abs(actual-expected[i]))))
    result={'passed':max(errors)==0.,'fresh_process':True,'checkpoint':str(a.checkpoint.resolve()),
        'learned_FP32_values_exact':True,'prediction_max_abs_errors':errors,
        'robot_SR_measured':False}
    atomic_json(a.output,result);print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Fresh native prediction differs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
