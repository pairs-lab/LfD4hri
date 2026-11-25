"""Measured native proprio and explicit image history for the new cohort.

Open and closure are independently captured native sensor fields. Inverting
an already-FP32 closure can differ from the measured FP32 open by1ULP; do not
replace the actual native observation by that lossy reconstruction.
"""
import hashlib
import numpy as np
from VLA.scene64_native_client import Scene64NativePolicy
from VLA.scene64_conditioning import model_input,PROMPT_SCHEMA
from VLA.privileged_native_trial import png


class Scene64R33Policy(Scene64NativePolicy):
    def reset(self,task):super().reset(task);self._previous=None
    def predict(self,obs):
        self.last_input=model_input(self.env,obs);prompt=self.last_input['prompt']
        current={'front_rgb_png':png(obs.front_rgb),'wrist_rgb_png':png(obs.wrist_rgb)}
        previous=self._previous or current
        native=np.r_[np.asarray(obs.joint_positions,np.float32),np.float32(obs.gripper_open),np.zeros(9,np.float32)].astype(np.float32)
        payload={'schema':PROMPT_SCHEMA,'instruction':obs.prompt,'scene64':self.last_input['scene64'].tolist(),
            'prompt':prompt,**current,'previous_front_rgb_png':previous['front_rgb_png'],
            'previous_wrist_rgb_png':previous['wrist_rgb_png'],'native_proprio':native.tolist(),
            'seed':self.seed,'inference_index':self.index}
        reply=self.session.post(self.url+'/act',json=payload,timeout=900);reply.raise_for_status();data=reply.json()
        if data.get('action_source')!='native_neural_model' or data.get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Actual neural source/full input identity missing')
        values=np.asarray(data['actions'],np.float32)
        if values.ndim!=2 or values.shape[1]!=7 or not len(values) or not np.isfinite(values).all():raise ValueError('Malformed native model output')
        self._previous=current;self.index+=1;self.last_native=values[0].copy();kind=data['action_kind']
        if kind=='absolute_joint_targets_and_closure':return self.env.joint_targets_to_canonical(values[0,:6],values[0,6])
        action=values[0].copy()
        if kind=='native_delta_ee_and_open_gripper':action[6]=1-2*action[6]
        elif kind!='canonical_delta_ee_7d':raise ValueError('Unrecognized native model action kind')
        return action
