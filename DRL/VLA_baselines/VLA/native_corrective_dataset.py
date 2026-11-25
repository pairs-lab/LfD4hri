"""Fresh native corrective data; exact replay admission is mandatory.

Provides the normal current-image OpenPI/OpenVLA view and real RDT image
history, including the pre-handoff student frame. This is not a trainer and
cannot turn a correction-only batch proof into a full-source qualification.
"""
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure,copy_requested_gripper_field
from VLA.scene64_conditioning import augment_prompt,encode_numeric
from VLA.scene64_dataset import nearest_joint_targets


class NativeCorrectiveDataset:
    def __init__(self,admission,action_horizon=5):
        admission=Path(admission).resolve();proof=json.loads(admission.read_text())
        if (not proof.get('passed') or proof.get('fresh_native_observation_action_view_qualified') is not True
            or proof.get('long_finetune_qualified') is not False or proof.get('episodes')!=38
            or proof.get('all40_original_attempts_preserved') is not True
            or proof.get('real_previous_camera_at_student_handoff') is not True
            or proof.get('prefix_actions_supervised') is not False):
            raise ValueError('Own complete fresh native corrective view is required')
        for path,digest in proof['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Native corrective admission changed: '+path)
        self.source=Path(proof['candidate']);self.admission=proof
        self.rows=json.loads((self.source/'index.json').read_text())
        self.horizon=int(action_horizon);self.pilot_only=True # correction-only NEVER authorizes full-source fitting
        if not 1<=self.horizon<=64:raise ValueError('Native horizon outside [1,64]')
        if [r['file'] for r in self.rows]!=[r['file'] for r in proof['rows']]:raise ValueError('Corrective row order changed')
        parts=[]
        for row in self.rows:
            if sha256_file(self.source/row['file'])!=row['payload_sha256']:raise ValueError('Native payload changed')
            with np.load(self.source/row['file'],allow_pickle=False) as data:
                parts.append(data['audit_pre_action_PRIVILEGED_state64'].copy())
        self.states=np.concatenate(parts);self.ends=np.cumsum([r['steps'] for r in self.rows]).tolist()
        if len(self.states)!=proof['frames']:raise ValueError('Corrective frame count changed')
    def __len__(self):return len(self.states)
    @lru_cache(maxsize=1)
    def episode(self,index):
        with np.load(self.source/self.rows[index]['file'],allow_pickle=False) as archive:
            result={key:archive[key] for key in ('front_rgb','wrist_rgb','joint_positions','gripper_closure',
                'next_joint_targets','next_gripper_closure','canonical_actions','prompt',
                'audit_previous_front_rgb','audit_previous_wrist_rgb')}
            copy_requested_gripper_field(archive,result)
        return result
    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<len(self):raise IndexError(index)
        ei=bisect_right(self.ends,index);fi=index-(self.ends[ei-1] if ei else 0)
        data=self.episode(ei);raw=self.states[index]
        ordinary=np.r_[data['joint_positions'][fi],data['gripper_closure'][fi]].astype(np.float32)
        np.testing.assert_allclose(ordinary,raw[:7],rtol=0,atol=1e-6)
        ids=np.minimum(fi+np.arange(self.horizon),self.rows[ei]['steps']-1)
        actions=np.c_[nearest_joint_targets(data['next_joint_targets'][ids],ordinary[:6]),
            native_gripper_closure(data)[ids]].astype(np.float32)
        canonical=data['canonical_actions'][ids].astype(np.float32,copy=True)
        opened=canonical.copy();opened[:,6]=(1-canonical[:,6])/2
        instruction=str(data['prompt'].item())
        return {'index':index,'episode_index':ei,'frame_index':fi,'image':data['front_rgb'][fi].copy(),
            'wrist_image':data['wrist_rgb'][fi].copy(),
            'previous_image':data['audit_previous_front_rgb'][fi].copy(),
            'previous_wrist_image':data['audit_previous_wrist_rgb'][fi].copy(),
            'joints':((ordinary[:6]+np.pi)%(2*np.pi)-np.pi).astype(np.float32),
            'gripper':ordinary[6:].copy(),'actions':actions,'canonical_actions':canonical,
            'language_actions':opened,'original_prompt':instruction,'prompt':augment_prompt(instruction,raw),
            'pre_action_scene64':raw.copy(),'numeric_scene67':encode_numeric(raw)}
    def __getitems__(self,indices):return [self[int(i)] for i in indices]


def rdt_dataset(data,stats,*,samples_per_episode=3,seed=20261015,action_horizon=10):
    """The REAL native sparse128 sample and unchanged7-DOF mask/loss targets."""
    from VLA.rdt.training_data import PyBulletRDTDataset
    from VLA.rdt.scene67_inputs import append_scene_conditioning
    class CorrectiveRDT(PyBulletRDTDataset):
        def __getitem__(self,index):
            sample=super().__getitem__(index)
            frames=sample.step_ids;payload=data.episode(index)
            start=data.ends[index-1] if index else 0
            current=data.states[start+frames]
            images=sample.images.copy()
            images[:,0]=payload['audit_previous_front_rgb'][frames]
            images[:,1]=payload['audit_previous_wrist_rgb'][frames]
            return replace(sample,images=images,states=append_scene_conditioning(sample.states,current,stats).numpy())
    return CorrectiveRDT(data.source,samples_per_episode=samples_per_episode,seed=seed,action_horizon=action_horizon)
