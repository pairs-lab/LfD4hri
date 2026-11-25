"""Unchanged locked evaluator, explicitly opt-in native67+GTcrop adapter."""
import argparse
import hashlib
from PIL import Image
import numpy as np
from VLA import evaluate_scene67_smolvla as native
from VLA.smolvla.target_crop import TargetCropSmolVLA,SCHEMA
from VLA.common.artifact_io import atomic_json


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=__import__('pathlib').Path,required=True)
    args,_=parser.parse_known_args()
    class ObservedCropPolicy(TargetCropSmolVLA):
        def build_sensor_batch(self,observation):
            batch=super().build_sensor_batch(observation)
            if observation.step%20==0:
                folder=args.output/'actual_policy_crop'/f"seed{self.seed}_{hashlib.sha256(observation.prompt.encode()).hexdigest()[:12]}"
                folder.mkdir(parents=True,exist_ok=True)
                image=batch['observation.images.camera3'][0].permute(1,2,0).numpy()
                Image.fromarray(np.rint(image*255).astype(np.uint8)).save(folder/f'step{observation.step:04d}.png')
                atomic_json(folder/f'step{observation.step:04d}.json',{'step':observation.step,'prompt':observation.prompt,
                    'privileged_policy_input':True,'before_action':True,'crop_receipt':self.last_input['target_crop_receipt']})
            return batch
    native.Scene67SmolVLA=ObservedCropPolicy
    def write(path,value):
        if path.name=='progress.json' and isinstance(value,dict) and value.get('model')=='smolvla_scene67':
            value={**value,'model':'smolvla_scene67_GT_target_crop','input_variant':SCHEMA,
                'privileged_GT_current_target_segmentation':True,'two_raw_views_retained':True}
        atomic_json(path,value)
    native.atomic_json=write
    native.main()


if __name__=='__main__':main()
