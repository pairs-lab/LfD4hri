"""Independent exact-weight/native three-view prediction reload gate."""
import argparse
from pathlib import Path

from VLA.stable_native_r44.conditioning import install
from .contract import require_visual


def main(args):
    cfg=require_visual(args.checkpoint);install()
    import VLA.new_vlas.reload_gate as native
    from .data import MapData
    from .runtime import create_runtime
    native.NativeData,native.create_runtime=MapData,create_runtime
    if not args.completed_fit:
        native.main(args)
        return
    # Native long mode needs a fresh prediction proof for the exact completed
    # resume checkpoint. Do not invent a gate.json for a long-fit checkpoint.
    import json
    import os
    import numpy as np
    import torch
    from VLA.common.artifact_io import atomic_json
    from VLA.common.provenance import sha256_file
    from VLA.new_vlas.train import wait_gpu
    out,cp=args.output.resolve(),args.checkpoint.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if (cfg['r53_lineage']['actual_extra_updates_with_third_GT_image']<1500
            or cfg['own_full_source_numeric_gate_not_final_model_SR']):
        raise ValueError('Completed actual1500+ long fit required')
    saved=json.loads((cp/'saved_values.json').read_text())
    for filename,key in [('trainable_masters.pt','trainable_master_sha256'),('optimizer.pt','optimizer_sha256'),
            ('reload_reference.npy','reference_prediction_sha256'),('config.json','config_sha256')]:
        if sha256_file(cp/filename)!=saved[key]:raise ValueError('Completed fit checkpoint bytes changed')
    torch.set_num_threads(2);torch.manual_seed(202643)
    data=MapData(cfg['data_root'])
    if sha256_file(data.root/'audit.json')!=cfg['data_audit_sha256']:raise ValueError('TRAIN source changed')
    wait_gpu(out)
    runtime=create_runtime(cfg['family'],data.stats)
    keys=runtime.load_trainables(cp/'trainable_masters.pt');runtime.to('cuda')
    actual=runtime.predict(data.sample(cfg['probe']['episode'],cfg['probe']['frame'],cfg['family']))
    reference=np.load(cp/'reload_reference.npy',allow_pickle=False)
    error=float(abs(actual-reference).max())
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or error>1e-6:
        raise ValueError('Completed3-view fresh-process prediction parity failed')
    atomic_json(out/'audit.json',{'passed':True,'family':cfg['family'],'checkpoint':str(cp),
        'data_audit_sha256':cfg['data_audit_sha256'],'FP32_trainable_values_exact':True,
        'trainable_keys':keys,'fresh_process_native_prediction_max_difference':error,
        'native_visual_receipt':runtime.r53_native_visual_receipt,'pid':os.getpid(),
        'completed_long_fit_reload_NOT_fake_gate2':True,'model_SR_established':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--completed-fit',action='store_true')
    main(parser.parse_args())
