"""Native VLA outputs unchanged apart from declared representation conversion."""
import hashlib
import numpy as np
import requests
from VLA.scene64_conditioning import model_input,PROMPT_SCHEMA
from VLA.privileged_native_trial import png


class Scene64NativePolicy:
    comparable=False
    def __init__(self,env,model,checkpoint,port,seed):
        self.env,self.model,self.seed=env,model,seed;self.url=f"http://127.0.0.1:{port}"
        self.session=requests.Session();self.session.trust_env=False
        reply=self.session.get(self.url+"/metadata",timeout=30);reply.raise_for_status();self.server_metadata=reply.json()
        meta=self.server_metadata
        if (meta.get("privileged_schema")!=PROMPT_SCHEMA or meta.get("actual_feedback_fields")!=64
            or meta.get("model")!=model or meta.get("checkpoint")!=str(checkpoint) or meta.get("oracle_or_ik_controller") is not False):
            raise ValueError("Need the actual newly qualified full-scene native server")
        self.index=0
    def reset(self,task):self.index=0
    def predict(self,obs):
        self.last_input=model_input(self.env,obs)
        prompt=self.last_input["prompt"]
        reply=self.session.post(self.url+"/act",json={"schema":PROMPT_SCHEMA,"instruction":obs.prompt,
            "scene64":self.last_input["scene64"].tolist(),"prompt":prompt,"front_rgb_png":png(obs.front_rgb),
            "wrist_rgb_png":png(obs.wrist_rgb),"seed":self.seed,"inference_index":self.index},timeout=900)
        reply.raise_for_status();data=reply.json()
        if data.get("action_source")!="native_neural_model" or data.get("prompt_sha256")!=hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError("Actual neural source/full input identity missing")
        values=np.asarray(data["actions"],np.float32)
        if values.ndim!=2 or values.shape[1]!=7 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Malformed native model command; no fallback")
        self.index+=1;self.last_native=values[0].copy();kind=data["action_kind"]
        if kind=="absolute_joint_targets_and_closure":return self.env.joint_targets_to_canonical(values[0,:6],values[0,6])
        action=values[0].copy()
        if kind=="native_delta_ee_and_open_gripper":action[6]=1-2*action[6]
        elif kind!="canonical_delta_ee_7d":raise ValueError("Unrecognized native action kind")
        return action
    def close(self):self.session.close()
