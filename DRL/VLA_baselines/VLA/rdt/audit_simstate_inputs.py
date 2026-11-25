"""Actual RDT+SimState dataset/collator/server tensor parity on FULL source.

CPU qualification only; not a model fit, inference performance or permission
to inject new inputs into an unchanged seven-sensor checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.rdt.adapter import RDTPolicy
from VLA.rdt.audit_checkpoint_predictions import observation_at
from VLA.rdt.simstate_inputs import RDTSimStateDataset, SCHEMA, EXTRA_SLOTS, prepare_privileged_sensor_tensors
from VLA.rdt.training_data import RDTBatchCollator, validate_language_embeddings, VALID_INDICES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--projection",type=Path,required=True)
    parser.add_argument("--embeddings",type=Path,required=True)
    parser.add_argument("--vision",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve actual model-specific input audits")
    torch.set_num_threads(2)
    data=RDTSimStateDataset(args.source,args.projection,action_horizon=10,samples_per_episode=4,seed=20261007)
    embeddings,_=validate_language_embeddings(args.embeddings,args.embeddings.with_suffix(".manifest.json"),args.source/"index.json",data.prompts)
    processor=AutoImageProcessor.from_pretrained(args.vision,local_files_only=True)
    collator=RDTBatchCollator(processor,embeddings);adapter=RDTPolicy(env=None,seed=20261007)
    errors={k:0. for k in ("images","states","state_elem_mask","lang_tokens","lang_attn_mask","ctrl_freqs")}
    rows=[]
    for task in ("touch","pick","move","pick_place"):
        numbers=[i for i,r in enumerate(data.index) if r["task"]["name"]==task]
        for number in [numbers[int(j)] for j in np.linspace(0,len(numbers)-1,8,dtype=int)]:
            samples=data[number];train=collator([samples])
            expected_mask=torch.zeros_like(train["state_elem_mask"]);expected_mask[:,list(VALID_INDICES)]=1
            if not torch.equal(train["state_elem_mask"],expected_mask) or torch.any(train["actions"][:,:,list(EXTRA_SLOTS)]!=0):
                raise ValueError("Additional observations became invalid extra motor actions")
            with np.load(args.source/samples.episode_file,allow_pickle=False) as episode:
                for row,frame in enumerate(samples.step_ids):
                    now=observation_at(episode,frame);before=observation_at(episode,max(0,frame-1))
                    payload=adapter.build_sensor_request(now,before)
                    payload.update(simstate_schema=SCHEMA,privileged_state18=data.projected_states[samples.episode_file][frame].tolist())
                    infer=prepare_privileged_sensor_tensors(payload,processor,embeddings,stats=data.simstate_stats)
                    for key in errors:
                        actual,expected=infer[key],train[key][row:row+1]
                        if actual.shape!=expected.shape: raise ValueError("Train/infer shape differs: "+key)
                        errors[key]=max(errors[key],float((actual.float()-expected.float()).abs().max()))
                    rows.append({"task":task,"episode":samples.episode_file,"frame":int(frame)})
            print(f"RDT privileged tensor samples:{len(rows)}/128",flush=True)
    passed=len(rows)==128 and all(v==0 for v in errors.values())
    atomic_json(args.output,{"passed":passed,"samples":len(rows),"rows":rows,"errors":errors,
        "source":str(args.source.resolve()),"source_index_sha256":sha256_file(args.source/"index.json"),
        "projection_audit_sha256":sha256_file(args.projection/"audit.json"),
        "input_implementation_sha256":sha256_file(Path(__file__).with_name("simstate_inputs.py")),
        "schema":SCHEMA,"privileged_model_inputs":True,"native_action_indices_unchanged":list(VALID_INDICES),
        "extra_observation_slots_NOT_action_DOFs":list(EXTRA_SLOTS),"GT_normalization":"TRAIN_only_zscore_extras11",
        "oracle_controller_at_deployment":False,"neural_weights_loaded":False,
        "long_fit_or_base_deployment_qualified":False,"model_success_established":False})
    if not passed: raise ValueError("Actual model-specific tensor parity failed")


if __name__=="__main__":main()
