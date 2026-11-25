"""Immutable parent weights/Adam/RNG; new observation recipe, not a fit."""
import argparse
import copy
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.stable_native_r44.conditioning import require_contract,add_contract
from .scene_map import contract
from .contract import source_hashes


def main(args):
    cp,out,data=args.parent.resolve(),args.output.resolve(),args.data.resolve()
    cfg=copy.deepcopy(require_contract(cp))
    proof=json.loads((data/'audit.json').read_text())
    if not proof['passed']:raise ValueError('Qualified full TRAIN data required')
    out.mkdir(parents=True,exist_ok=False)
    saved=json.loads((cp/'saved_values.json').read_text())
    for name,key in [('trainable_masters.pt','trainable_master_sha256'),('optimizer.pt','optimizer_sha256')]:
        if sha256_file(cp/name)!=saved[key]:raise ValueError('Parent bytes changed')
        os.link(cp/name,out/name)
    os.link(cp/'torch_rng.pt',out/'torch_rng.pt')
    new_stats=json.loads((data/'stats.json').read_text())
    changed=cfg['stats']!=new_stats
    step=cfg['actual_optimizer_updates']
    cfg.update(data_root=str(data),stats=new_stats,data_audit_sha256=sha256_file(data/'audit.json'),
        model_input='two_UNCHANGED_real_RGB_plus_THIRD_causal_simGT_image_native_proprio_ALL67',
        WARM_START_ONLY_DO_NOT_SERVE_OR_REPORT_AS_NEW_FIT=True,
        privileged_visual_observation=contract(),own_full_source_numeric_gate_not_final_model_SR=True,
        r53_lineage={'parent_checkpoint':str(cp),'parent_actual_optimizer_updates':step,
            'actual_extra_updates_with_third_GT_image':0,'normalizers_changed_explicitly':changed,
            'all_weight_Adam_RNG_values_unchanged':True})
    cfg['code'].update(source_hashes())
    atomic_json(out/'config.json',cfg)
    saved.update(config_sha256=sha256_file(out/'config.json'),reference_prediction_sha256=None,
        shape=None,finite_prediction=None,WARM_START_ONLY_no_new_prediction_reference=True)
    atomic_json(out/'saved_values.json',saved)
    add_contract(out,parent=cp)
    atomic_json(out/'migration.json',{'parent':str(cp),'actual_new_optimizer_updates':0,
        'servable':False,'changed_TRAIN_normalizer':changed,'required':'gate2_fresh_reload_then1500_total_added_updates'})
    print(json.dumps({'warm_start':str(out),'parent_updates':step,'new_updates':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
