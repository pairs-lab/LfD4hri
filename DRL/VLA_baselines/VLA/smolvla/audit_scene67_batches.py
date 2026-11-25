"""REAL native temporal loader, including appended on-policy corrections."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main():
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import PolicyFeature,FeatureType
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from VLA.smolvla.scene67_dataset import Scene67Dataset
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base-model","projection","recovery-admission","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("Retain every native batch audit")
    cfg=TrainPipelineConfig.from_pretrained(str(a.base_model));cfg.policy.device="cpu";cfg.policy.max_state_dim=67
    if cfg.policy.chunk_size!=5 or cfg.policy.observation_delta_indices!=[0] or cfg.dataset.image_transforms.enable:
        raise ValueError("Exact current image /H5 recipe required")
    base=make_dataset(cfg);data=Scene67Dataset(base,a.projection,a.recovery_admission)
    ids=np.r_[np.linspace(0,len(base)-1,96,dtype=int),np.linspace(len(base),len(data)-1,32,dtype=int)].tolist()
    cfg.policy.input_features["observation.state"]=PolicyFeature(type=FeatureType.STATE,shape=(67,))
    stats={cfg.rename_map.get(k,k):v for k,v in data.meta.stats.items()}
    proc,_=make_pre_post_processors(cfg.policy,pretrained_path=None,dataset_stats=stats,
        preprocessor_overrides={"device_processor":{"device":"cpu"}})
    maximum=0.;rows=[]
    for number,batch in enumerate(DataLoader(data,batch_size=2,sampler=ids,num_workers=0)):
        selected=ids[2*number:2*number+2];single=default_collate([data[i] for i in selected])
        for key in ("observation.state","action",*data.meta.camera_keys):
            torch.testing.assert_close(batch[key],single[key],rtol=0,atol=0)
        actual=_preprocess_dataset_batch(copy.deepcopy(batch),data.meta.camera_keys,cfg.rename_map,proc)
        state=SmolVLAPolicy.prepare_state(SimpleNamespace(config=cfg.policy),actual)
        expected=(torch.from_numpy(data.states[selected])-torch.as_tensor(stats["observation.state"]["mean"]))/(
            torch.as_tensor(stats["observation.state"]["std"])+1e-8)
        if state.shape!=(2,67) or not torch.isfinite(state).all():raise ValueError("Native input was truncated")
        error=float((state-expected).abs().max());maximum=max(maximum,error)
        if error>1e-5:raise ValueError("Native normalizer differs")
        rows.append({"indices":selected,"error":error})
    atomic_json(a.output,{"passed":True,"actual_native_batches":len(rows),"samples":len(ids),"corrective_samples":32,
        "normalization_max_error":maximum,"total_frames":len(data),"original_frames":len(base),
        "input_shape":[67],"native_padding_dim":67,"rows":rows,"neural_weights_loaded":False,
        "model_SR_established":False,"normalization_contract":data.normalization_contract,
        "projection_audit_sha256":sha256_file(a.projection/"audit.json"),
        "recovery_admission_sha256":sha256_file(a.recovery_admission)})


if __name__ == "__main__":main()
