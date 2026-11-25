"""Exact native previous-requested-pose labels, causal7 and TRAIN-only scales."""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from VLA.refinement_r99.data import Data as ParentData
from VLA.new_vlas.action_contracts import rotation_matrix,canonical_from_absolute20


def poses(native20):
    values=np.asarray(native20,np.float64)
    matrix=np.broadcast_to(np.eye(4),(len(values),4,4)).copy()
    matrix[:,:3,:3]=rotation_matrix(values[:,3:9][:,[0,2,4,1,3,5]])
    matrix[:,:3,3]=values[:,:3]
    return matrix


def native_labels(number):
    current=poses(number['state20']);target=poses(number['action20'])
    anchor=np.concatenate((current[:1],target[:-1]),axis=0)
    rotation=np.swapaxes(anchor[:,:3,:3],1,2)@target[:,:3,:3]
    translation=np.einsum('nij,nj->ni',np.swapaxes(anchor[:,:3,:3],1,2),target[:,:3,3]-anchor[:,:3,3])
    action=np.c_[translation,Rotation.from_matrix(rotation).as_euler('xyz'),-number['canonical7'][:,6]]
    state=np.c_[current[:,:3,3],Rotation.from_matrix(current[:,:3,:3]).as_euler('xyz'),
        (number['scene64'][:,6]<.5).astype(np.float64)]
    if not np.isfinite(action).all() or not np.isfinite(state).all():raise ValueError('Nonfinite native Seer labels/state')
    # Independent canonical reconstruction includes actual shared plant clamps.
    decoded_target=anchor.copy();decoded_target[:,:3,:3]=anchor[:,:3,:3]@Rotation.from_euler('xyz',action[:,3:6]).as_matrix()
    decoded_target[:,:3,3]=anchor[:,:3,3]+np.einsum('nij,nj->ni',anchor[:,:3,:3],action[:,:3])
    if np.max(abs(decoded_target-target))>1e-6:raise ValueError('Native previous-command pose conversion mismatch')
    return {'local_action':action.astype(np.float32),'native_state':state.astype(np.float32),
        'current_pose':current,'target_pose':target,'previous_command':anchor}


def statistics(data):
    max_action=np.zeros(6);counts={};nframes=0
    for ei,row in enumerate(data.rows):
        values=native_labels(data.numeric(ei));a=values['local_action']
        max_action=np.maximum(max_action,np.max(abs(a[:,:6]),axis=0))
        counts[row['task']]=counts.get(row['task'],0)+len(a);nframes+=len(a)
    # Preserve native shared xyz and shared Euler scalar conventions, but make
    # the bounded Tanh head represent ALL actual TRAIN labels without clipping.
    scales=np.r_[[max(.02,float(max_action[:3].max())*1.05)]*3,
                 [max(.05,float(max_action[3:].max())*1.05)]*3]
    return {'scene67':data.stats['scene67'],'local_action_scale6':scales.tolist(),
        'observed_TRAIN_max_abs6':max_action.tolist(),'frames':nframes,'per_action_frames':counts,
        'native_history':7,'native_horizon':3,'future_image_offset':3,
        'gripper':'native action+1 OPEN, -1 CLOSE; BCE target (native+1)/2',
        'action_anchor':'previous requested clamped pose; reset anchor is measured pose',
        'scales_derived_from_TRAIN_only_no_silent_label_clipping':True}


class Data(ParentData):
    @lru_cache(maxsize=4)
    def converted(self,ei):return native_labels(self.numeric(ei))

    def native_sample(self,ei,fi):
        n=self.numeric(ei);rgb=self.rgb(ei);row=self.rows[ei];v=self.converted(ei);length=row['frames']
        history=np.maximum(np.arange(fi-6,fi+1),0)
        future=history[:,None]+np.arange(3)[None]
        image_future=history+3
        return {'front_history':rgb['front_rgb'][history].copy(),'wrist_history':rgb['wrist_rgb'][history].copy(),
            'state_history':v['native_state'][history].copy(),'scene64_history':n['scene64'][history].copy(),
            'instruction':row['instruction'],'local_actions':v['local_action'][np.minimum(future,length-1)].copy(),
            'valid_actions':future<length,'future_front_labels':rgb['front_rgb'][np.minimum(image_future,length-1)].copy(),
            'future_wrist_labels':rgb['wrist_rgb'][np.minimum(image_future,length-1)].copy(),
            'valid_future_images':image_future<length,'history_indices':history.tolist(),
            'current_pose':v['current_pose'][fi].copy(),'previous_command':v['previous_command'][fi].copy(),
            'episode':ei,'frame':fi,'task':row['task']}
