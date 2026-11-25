"""Exercise EVERY admitted correction through the real native batch path.

This can run before the full physical projection finishes. It does not
pretend to construct a complete Scene67Dataset or qualify a long fit.
Only its existing corrective fetch methods are exercised with real source
templates and real corrective states; no synthetic base states are supplied.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import NUMERIC_NAMES, encode_numeric


def main():
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # register native config
    from lerobot.datasets.factory import make_dataset
    from VLA.smolvla.scene67_dataset import Scene67Dataset
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base-model","admission","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("Preserve every audit attempt")
    proof=json.loads(a.admission.read_text())
    if not proof.get("passed") or "smolvla" not in proof["allowed_models"]:raise ValueError("Exact correction admission required")
    for name,digest in proof["dependencies"].items():
        if sha256_file(name)!=digest:raise ValueError("Admission changed")
    cfg=TrainPipelineConfig.from_pretrained(str(a.base_model));cfg.policy.device="cpu"
    if cfg.policy.chunk_size!=5 or cfg.policy.observation_delta_indices!=[0] or cfg.dataset.image_transforms.enable:
        raise ValueError("Current-image/H5/native action recipe required")
    base=make_dataset(cfg);source=Path(json.loads((Path(proof['candidate'])/'plan.json').read_text())["source"])
    parent_rows=json.loads((source/'index.json').read_text())
    offsets={};total=0
    for row in parent_rows:offsets[row['file']]=total;total+=row['steps']
    if total!=len(base):raise ValueError("Real native source count differs")
    root=Path(proof['candidate']);by_file={r['file']:r for r in json.loads((root/'index.json').read_text()) if r.get('file')}
    data=object.__new__(Scene67Dataset)
    data.base=base;data.original_length=len(base);data.parent_rows=parent_rows;data.parent_offsets=offsets
    data.corrections=[];data.correction_ends=[];pieces=[];count=0
    for item in proof['rows']:
        row=by_file[item['file']];path=root/item['file']
        if sha256_file(path)!=item['payload_sha256']:raise ValueError("Correction payload changed")
        with np.load(path,allow_pickle=False) as z:states=z['audit_pre_action_PRIVILEGED_state64'].copy()
        pieces.append(np.asarray([encode_numeric(v) for v in states],np.float32));count+=len(states)
        data.corrections.append({**row,'path':str(path)});data.correction_ends.append(count)
    encoded=np.concatenate(pieces)
    class CorrectionOnlyStates:
        def __getitem__(self,index):
            if not data.original_length<=index<data.original_length+len(encoded):
                raise IndexError("No base state fabricated by this correction-only audit")
            return encoded[index-data.original_length]
        def __len__(self):return data.original_length+len(encoded)
    data.states=CorrectionOnlyStates();data.meta=copy.deepcopy(base.meta)
    data.meta.info.features['observation.state']={'dtype':'float32','shape':[67],'names':NUMERIC_NAMES}
    ids=[];previous=0
    for end in data.correction_ends:
        ids.extend(data.original_length+previous+i for i in sorted({0,(end-previous)//2,end-previous-1}));previous=end
    checked=[]
    for number,batch in enumerate(DataLoader(data,batch_size=2,sampler=ids,num_workers=0)):
        selected=ids[number*2:number*2+2];indexed=default_collate([data[i] for i in selected])
        for key in ('observation.state','action','action_is_pad',*data.meta.camera_keys):
            if key in batch:torch.testing.assert_close(batch[key],indexed[key],rtol=0,atol=0)
        if batch['observation.state'].shape!=(len(selected),1,67) or batch['action'].shape!=(len(selected),5,7):
            raise ValueError("Native corrective tensor shapes differ")
        for i in selected:
            local=i-data.original_length;ei=int(np.searchsorted(data.correction_ends,local,side='right'))
            fi=local-(data.correction_ends[ei-1] if ei else 0);row=data.corrections[ei]
            with np.load(row['path'],allow_pickle=False) as z:
                single=data[i];n=len(z['canonical_actions']);stepids=fi+np.arange(5)
                torch.testing.assert_close(single['action'],torch.from_numpy(z['canonical_actions'][np.minimum(stepids,n-1)]),rtol=0,atol=0)
                if 'action_is_pad' in single:torch.testing.assert_close(single['action_is_pad'],torch.from_numpy(stepids>=n),rtol=0,atol=0)
                for camera in data.meta.camera_keys:
                    role=camera.rsplit('.',1)[1];expected=torch.from_numpy(z[role+'_rgb'][fi].copy()).permute(2,0,1).float()/255
                    if single[camera].ndim==4: expected=expected.unsqueeze(0)
                    torch.testing.assert_close(single[camera],expected,rtol=0,atol=0)
                    if camera+'_is_pad' in single and torch.any(single[camera+'_is_pad']):raise ValueError('Current corrective camera incorrectly marked as padded')
                if 'observation.state_is_pad' in single and torch.any(single['observation.state_is_pad']):raise ValueError('Current corrective state incorrectly padded')
                if single['task']!=str(z['prompt'].item()):raise ValueError("Corrective text/target changed")
                if int(single['episode_index'])!=len(parent_rows)+ei:raise ValueError("Corrective episode remapped")
        checked.append({'indices':selected,'passed':True})
    atomic_json(a.output,{'passed':True,'scope':'real_corrective_native_batches_ONLY_not_full_fit_qualification',
        'correction_episodes':len(data.corrections),'correction_frames':len(encoded),'samples':len(ids),
        'actual_loader_batches':len(checked),'all_correction_episodes_checked':True,'synthetic_base_states':False,
        'pixel_action_text_pad_index_max_errors':0,'long_training_qualified':False,'neural_weights_loaded':False,
        'admission_sha256':sha256_file(a.admission),'fetch_implementation_sha256':sha256_file(Path(__file__).with_name('scene67_dataset.py')),
        'rows':checked})


if __name__=='__main__':main()
