"""Own actual saved FP32 pi0 weights and deterministic native8 GPU gate."""
import argparse
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import checkpoint_contract
from .data import Data,anchors


def main(a):
    from .runtime_pi0 import Runtime
    cfg=checkpoint_contract(a.checkpoint);data=Data(cfg['data']);native=Runtime(a.checkpoint)
    errors=[]
    for ei,fi in anchors(data):
        x=data.at(ei,fi)
        one=native.predict(x,202654);two=native.predict(x,202654)
        errors.append(float(abs(one-two).max()))
    if len(errors)!=8 or max(errors)>1e-6:raise ValueError('Native seeded pi0 repeatability failed')
    atomic_json(a.output,{'passed':True,'model':'pi0','checkpoint':str(a.checkpoint.resolve()),
        'actual_optimizer_updates':cfg['added_optimizer_updates'],'all_saved_learned_values_exact':native.exact_values,
        'actual_neural_predictions':8,'additional_repeat_predictions':8,'max_repeat_action_error':max(errors),
        'bound_files':{str(a.checkpoint/'r54_contract.json'):sha256_file(a.checkpoint/'r54_contract.json'),
                       cfg['precision']:cfg['precision_sha256']},'model_SR_established':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
