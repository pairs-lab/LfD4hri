"""Run REAL native config/dataset/sampler entry; stop BEFORE model allocation.

Only the heavy policy class is replaced by a CPU inspection callback AFTER
the real native factory infers and renames its dataset features. This is not a
model/gradient/CUDA qualification and cannot authorize long fitting alone.
"""
import argparse
import os
import sys
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_target_crop_r36 import fit_command
from VLA.smolvla.target_crop import CAMERA,NATIVE_CAMERA,SCHEMA


class Inspected(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('warmstart','stub-fit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.stub_fit.exists():raise FileExistsError('Keep failed native-entry attempts')
    from VLA.smolvla import train_target_crop,balanced_sampling,train_fp32
    observed={}
    def inspect(*args,**kwargs):
        cfg=kwargs.get('config',args[0] if args else None);meta=kwargs.get('dataset_meta')
        if cfg is None or meta is None:raise ValueError('Actual native policy constructor contract missing')
        if tuple(cfg.input_features['observation.state'].shape)!=(67,) or NATIVE_CAMERA not in cfg.input_features:
            raise ValueError('Actual parsed native policy lost state67/third camera')
        if CAMERA not in meta.camera_keys or len(meta.camera_keys)!=3:raise ValueError('Actual dataset metadata lost target view')
        weights=balanced_sampling.frame_weights(None,.1,target_balanced=True)
        if len(weights)!=116257 or abs(float(weights.sum())-1)>1e-10:raise ValueError('Native original motion mass differs')
        observed.update(passed=True,schema=SCHEMA,actual_parsed_state_dimension=67,
            actual_native_raw_camera_keys=list(meta.camera_keys),actual_native_feature_keys=list(cfg.input_features),
            original_motion_weights=len(weights),optimizer_lr=cfg.optimizer_lr,
            neural_model_allocated=False,optimizer_updates=0,
            native_factory_feature_inference_and_rename_exercised=True,
            heavy_native_model_entry_replaced_by_inspection_callback=True,
            purpose='CPU_native_trainer_wiring_NOT_CUDA_qualification',
            implementation_sha256=sha256_file(Path(__file__)))
        raise Inspected()
    class InspectionPolicy:
        def __init__(self,*args,**kwargs):inspect(*args,**kwargs)
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return inspect(*args,**kwargs)
    train_fp32.fp32_policy_class=lambda *_a,**_k:InspectionPolicy
    os.environ.pop('VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB',None)
    command=fit_command(a.warmstart,a.stub_fit,2)[3:]+['--policy.device=cpu']
    old=sys.argv
    try:
        sys.argv=[old[0],*command]
        try:train_target_crop.main()
        except Inspected:pass
        if not observed:raise ValueError('Native trainer did not reach actual policy construction')
        atomic_json(a.output,observed)
    finally:sys.argv=old


if __name__=='__main__':main()
