"""Native SmolVLA fitting with GT target view and TRAIN-only motion balancing."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.common.artifact_io import atomic_json
from VLA.smolvla.target_crop import TargetCropDataset,SCHEMA


def main():
    from VLA.smolvla import scene67_dataset,balanced_sampling,train_fp32,train_scene67
    from VLA.openvla.recovery_balanced import sampling_weights
    gate=json.loads(Path(os.environ['VLA_TARGET_CROP_GATE']).read_text())
    if not gate.get('passed') or gate['schema']!=SCHEMA or gate['samples']!=128:raise ValueError('Own real native3-view gate required')
    for path,digest in gate['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Own GTcrop input gate changed')
    if int(os.environ['VLA_SCENE67_UPDATES'])>2:
        proof=json.loads(Path(os.environ['VLA_SCENE67_SMOKE_QUALIFICATION']).read_text())
        if proof.get('target_crop_schema')!=SCHEMA:raise ValueError('An old two-camera smoke cannot qualify this fit')
    created=[]
    class ActualCropData(TargetCropDataset):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            class View:
                rows=self.parent_rows;states=self.raw64[:self.original_length];ends=self.ends;source=Path(self.audit['source'])
                def __len__(_self):return self.original_length
            weights,contract=sampling_weights(View(),SimpleNamespace(rows=self.corrections))
            self.motion_weights=weights[:self.original_length]/.8
            self.motion_contract={**contract,'corrective_sampling':'native Smol task/episode-balanced20%; original task/target/motion-balanced80%'}
            created.append(self)
            atomic_json(Path(os.environ['VLA_SMOLVLA_MASTER_AUDIT']).with_name('actual_target_crop_sampling.json'),self.motion_contract)
    scene67_dataset.Scene67Dataset=ActualCropData
    def motion_weights(*_args,**_kwargs):
        if len(created)!=1:raise ValueError('Actual source must exist before native sampling')
        return created[0].motion_weights.copy()
    balanced_sampling.frame_weights=motion_weights
    original_fp32=train_fp32.main
    def native_train():
        balanced_sampling.sampling_contract=lambda *_a,**_k:{'profile':SCHEMA,
            'original_mass':.8,'corrective_mass':.2,'original_motion_sampling':'task/target/label-stratum balanced; actual counts in actual_target_crop_sampling.json',
            'third_view_current_GT_only':True,
            'source_images_and_action_labels_changed':False,'future_or_teacher_actions_in_input':False}
        original_fp32()
    train_fp32.main=native_train
    return train_scene67.main()


if __name__=='__main__':main()
