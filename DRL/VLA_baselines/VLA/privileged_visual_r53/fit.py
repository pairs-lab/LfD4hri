"""Same native loss/FP32 optimizer and R52 sampler, with a third GT image."""
import argparse
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from .contract import source_hashes
from .scene_map import contract


def main(args):
    import VLA.new_vlas.dataset as dataset
    import VLA.new_vlas.train as native
    from .data import MapData
    from .runtime import create_runtime
    from VLA.fit_feedback_r52 import main as fit
    proof=json.loads(args.input_proof.read_text())
    if (not proof['passed'] or proof['family']!=args.family or proof['profile']!=contract()
            or proof['data_audit_sha256']!=sha256_file(args.data/'audit.json')):
        raise ValueError('Own actual third-image CPU input proof required before GPU gate')
    parent=json.loads((args.resume/'config.json').read_text())
    origin=parent.get('r53_lineage',{}).get('parent_actual_optimizer_updates',parent['actual_optimizer_updates'])
    dataset.NativeData=MapData
    native.NativeData=MapData
    native.create_runtime=create_runtime
    original=native.atomic_json
    def atomic(path,value):
        if Path(path).name in ('plan.json','config.json'):
            value={**value,'privileged_visual_observation':contract(),
                'model_input':'two_UNCHANGED_real_RGB_plus_THIRD_causal_simGT_image_native_proprio_ALL67',
                'input_proof':str(args.input_proof.resolve()),'input_proof_sha256':sha256_file(args.input_proof),
                'r53_lineage':{'parent_actual_optimizer_updates':origin,
                    'actual_extra_updates_with_third_GT_image':max(0,value.get('actual_optimizer_updates',origin)-origin),
                    'base_loss_action_semantics_and_controller_unchanged':True}}
            if Path(path).name=='config.json':value['code'].update(source_hashes())
        original(path,value)
    native.atomic_json=atomic
    fit(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('xvla','vla_adapter'),required=True)
    parser.add_argument('--mode',choices=('gate','long'),required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,required=True)
    parser.add_argument('--total-updates',type=int,required=True)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--reload-proof',type=Path)
    parser.add_argument('--input-proof',type=Path,required=True)
    a=parser.parse_args()
    start=json.loads((a.resume/'config.json').read_text())['actual_optimizer_updates']
    if a.mode=='gate' and (a.updates-start!=2 or a.accumulation!=4):parser.error('Two new real updates /eight micros required')
    main(a)
