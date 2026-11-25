"""Lossless native OpenPI restore; no extra neural head or action repair."""
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import checkpoint_contract, read
from .pi0 import build_config,pi_sample


class Runtime:
    def __init__(self,checkpoint):
        from flax import nnx
        from flax.traverse_util import flatten_dict
        from openpi.models import model as model_api
        from openpi.policies import policy_config
        self.contract=checkpoint_contract(checkpoint)
        from VLA.common.provenance import sha256_file
        proof=read(self.contract['precision'])
        if sha256_file(self.contract['precision'])!=self.contract['precision_sha256'] or not proof['passed']:
            raise ValueError('Actual Orbax precision proof changed')
        self.cfg=build_config(self.contract['stats'],checkpoint,self.contract['requested_updates'])
        restore=model_api.restore_params
        def exact(path,*args,**kwargs):
            if Path(path).resolve()!=(checkpoint/'params').resolve():raise ValueError('Unexpected weight restore')
            kwargs['dtype']=None;return restore(path,*args,**kwargs)
        with patch.object(model_api,'restore_params',exact):
            self.native=policy_config.create_trained_policy(self.cfg,str(checkpoint))
        actual=flatten_dict(nnx.state(self.native._model).filter(self.cfg.trainable_filter).to_pure_dict())
        if set(actual)!={tuple(r['path']) for r in proof['trainable_tensors']}:raise ValueError('Learned tensor set changed')
        for r in proof['trainable_tensors']:
            v=np.asarray(actual[tuple(r['path'])])
            if str(v.dtype)!='float32' or hashlib.sha256(v.tobytes()).hexdigest()!=r['saved_sha256']:
                raise ValueError('Saved pi0 learned value changed on fresh native reload')
        self.exact_values=True

    def predict(self,sample,seed):
        raw=pi_sample(sample);raw.pop('actions',None)
        prepared=self.native._input_transform(dict(raw))
        if (len(prepared['image'])!=3 or not all(bool(v) for v in prepared['image_mask'].values())
            or int(np.asarray(prepared['tokenized_prompt_mask']).sum())>=self.cfg.model.max_token_len):
            raise ValueError('Actual native pi0 observation missing/truncated')
        noise=np.random.default_rng(int(seed)).standard_normal((5,self.cfg.model.action_dim),dtype=np.float32)
        result=np.asarray(self.native.infer(raw,noise=noise)['actions'],np.float32)
        if result.shape!=(5,7) or not np.isfinite(result).all():raise ValueError('Malformed native pi0 output')
        result[:,6]=2*np.clip(result[:,6],0,1)-1
        return result
