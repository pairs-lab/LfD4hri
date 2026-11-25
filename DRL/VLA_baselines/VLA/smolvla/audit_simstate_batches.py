"""Exercise the REAL training DataLoader/temporal factory, not individual rows.

Also compare the ordinary state7 native batched path against single-row reads,
so a privileged-wrapper defect cannot be blamed for historical plain-VLA SR.
No neural model/GPU weights or target/action controller is loaded.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.smolvla.simstate_dataset import SimStateDataset


def main():
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection",type=Path,required=True)
    parser.add_argument("--base-model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Keep every actual batch audit")
    torch.set_num_threads(2)
    cfg = TrainPipelineConfig.from_pretrained(str(args.base_model))
    cfg.policy.device="cpu"
    if cfg.policy.observation_delta_indices != [0] or cfg.policy.chunk_size != 5:
        raise ValueError("Expected the exact current-observation / H5 training recipe")
    if cfg.dataset.image_transforms.enable: raise ValueError("Exact audit requires original no-augmentation recipe")
    base = make_dataset(cfg)
    data = SimStateDataset(base,args.projection)
    selected = np.linspace(0,len(data)-1,128,dtype=int).tolist()
    # Verify actual ORIGINAL native batch fetching, with its timestamp windows.
    stock,_ = make_pre_post_processors(cfg.policy,str(args.base_model),
        preprocessor_overrides={"device_processor":{"device":"cpu"}})
    view7 = SimpleNamespace(config=copy.deepcopy(cfg.policy))
    baseline_max={"state":0.,"action":0.,"image":0.}
    for offset,batch in enumerate(DataLoader(base,batch_size=2,sampler=selected,num_workers=0)):
        ids=selected[offset*2:offset*2+2]
        single=default_collate([base[i] for i in ids])
        native=_preprocess_dataset_batch(batch,base.meta.camera_keys,cfg.rename_map,stock)
        indexed=_preprocess_dataset_batch(single,base.meta.camera_keys,cfg.rename_map,stock)
        for label,method in (("state",SmolVLAPolicy.prepare_state),("action",SmolVLAPolicy.prepare_action)):
            a,b=method(view7,native),method(view7,indexed)
            if a.shape!=b.shape: raise ValueError("Ordinary native-batch shape drift: "+label)
            baseline_max[label]=max(baseline_max[label],float((a-b).abs().max()))
        actual_images,_=SmolVLAPolicy.prepare_images(view7,native)
        expected_images,_=SmolVLAPolicy.prepare_images(view7,indexed)
        baseline_max["image"]=max(baseline_max["image"],max(float((a-b).abs().max()) for a,b in zip(actual_images,expected_images,strict=True)))
    if any(v!=0 for v in baseline_max.values()):
        raise ValueError("Ordinary native batch path differs; inspect before attributing model failure")
    cfg.policy.input_features["observation.state"]=PolicyFeature(type=FeatureType.STATE,shape=(18,))
    mapping=cfg.rename_map
    stats={mapping.get(k,k):v for k,v in data.meta.stats.items()}
    processor,_=make_pre_post_processors(cfg.policy,pretrained_path=None,dataset_stats=stats,
        preprocessor_overrides={"device_processor":{"device":"cpu"}})
    view18=SimpleNamespace(config=cfg.policy)
    rows=[]; maximum=0.
    for number,batch in enumerate(DataLoader(data,batch_size=2,sampler=selected,num_workers=0)):
        ids=selected[number*2:number*2+2]
        if batch["observation.state"].shape[-1]!=18:
            raise ValueError("Actual DataLoader bypassed the state18 wrapper")
        processed=_preprocess_dataset_batch(copy.deepcopy(batch),data.meta.camera_keys,mapping,processor)
        state=SmolVLAPolicy.prepare_state(view18,processed)
        mean,std=stats["observation.state"]["mean"],stats["observation.state"]["std"]
        expected=(torch.from_numpy(data.states[ids])-torch.as_tensor(mean))/(torch.as_tensor(std)+1e-8)
        if state.shape!=(len(ids),32) or torch.any(state[:,18:]!=0) or not torch.isfinite(state).all():
            raise ValueError("Actual batch state18 normalization/padding failed")
        error=float((state[:,:18]-expected).abs().max());maximum=max(maximum,error)
        if error>1e-5: raise ValueError("Batch uses stale seven-dimensional normalization")
        # Images and supervised action chunks must be exactly untouched.
        original=default_collate([base[i] for i in ids])
        for key in (*data.meta.camera_keys,"action"):
            torch.testing.assert_close(batch[key],original[key],rtol=0,atol=0)
        rows.append({"indices":ids,"raw_batch_state_shape":list(batch["observation.state"].shape),"normalization_error":error})
    atomic_json(args.output,{"passed":True,"samples":128,"actual_batches":len(rows),"rows":rows,
        "actual_temporal_training_dataset_factory":True,"torch_DataLoader_batched_fetch_exercised":True,
        "ordinary_state7_native_batch_vs_indexed_max_errors":baseline_max,
        "state18_batch_normalization_max_abs_error":maximum,"native_padding_dim":32,
        "source_images_action_chunks_untouched":True,"neural_weights_loaded":False,
        "projection_audit_sha256":sha256_file(args.projection/"audit.json"),
        "wrapper_sha256":sha256_file(Path(__file__).with_name("simstate_dataset.py")),
        "base_train_config_sha256":sha256_file(args.base_model/"train_config.json"),
        "model_success_established":False})


if __name__=="__main__":main()
