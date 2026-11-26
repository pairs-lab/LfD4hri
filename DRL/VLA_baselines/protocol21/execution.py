"""Native action decoding and H/K scheduling, without task-conditioned corrections."""
from collections import deque
import inspect
import numpy as np
from .specs import SPECS


def prediction(runtime, family, sample, stats, seed=0):
    # Offline samples carry supervised labels; never forward them to inference.
    keys={'front','wrist','instruction','raw_state21','state','ee_position','time_s',
          'history_times','prompt','history28','history_images','state_history',
          'robot21_history','front_history','wrist_history','depth_z_m','intrinsics','extrinsics_OpenCV'}
    sample={k:v for k,v in sample.items() if k in keys}
    arguments = inspect.signature(runtime.predict).parameters
    result = runtime.predict(sample,seed=seed) if 'seed' in arguments else runtime.predict(sample)
    result = np.asarray(result,np.float32)
    width = 20 if family == 'xvla' else 7
    if result.shape != (SPECS[family].horizon,width) or not np.isfinite(result).all():
        raise ValueError('Malformed actual native prediction')
    # Adapter's native API already unnormalizes actions using pybullet_train.
    if family == 'vla_adapter':
        from VLA.new_vlas.policy_outputs import native_gripper_command
        result[:,6] = [native_gripper_command(v,convention='adapter_open01') for v in result[:,6]]
    return result


class Executor:
    """Caller pushes one measured Frame after EACH action before calling step()."""
    def __init__(self, family, runtime, stats):
        from .inputs import Inputs
        self.family,self.runtime,self.stats = family,runtime,stats
        self.inputs = Inputs(family,stats)
        self.queue = deque()
        self.previous_command = None
        self.queries = 0
        self._pending_frame = None

    def reset(self):
        self.inputs.reset(); self.queue.clear(); self.previous_command=None; self.queries=0; self._pending_frame=None
        model = getattr(self.runtime,'model',None)
        if hasattr(model,'reset'): model.reset()

    def step(self, frame):
        from .inputs import rotation
        from VLA.new_vlas.action_contracts import canonical_from_absolute20,absolute20_from_canonical
        from VLA.admission_r100.seer_data import poses
        if self.family=='seer' and self._pending_frame is not None:
            raise ValueError('Call record_execution(info) after env.step; Seer must anchor to the actual safety-clamped request')
        self.inputs.push(frame)
        if self.previous_command is None:
            self.previous_command = np.eye(4)
            self.previous_command[:3,:3] = rotation(frame.raw_state21)
            self.previous_command[:3,3] = frame.ee_position
        if not self.queue:
            s = self.inputs.sample()
            raw = prediction(self.runtime,self.family,s,self.stats,self.queries)
            self.queue.extend(raw[:SPECS[self.family].execute])
            self.queries += 1
        action = self.queue.popleft().copy()
        if self.family == 'xvla':
            # Decode every executed absolute prediction against NOW's measured pose.
            from VLA.new_vlas.policy_outputs import native_gripper_command
            grip=native_gripper_command(action[9],convention='xvla_closed01')
            action = canonical_from_absolute20(action,frame.ee_position,frame.raw_state21[11:17])
            action[6]=grip
        elif self.family == 'seer':
            from scipy.spatial.transform import Rotation
            local = np.eye(4); local[:3,:3] = Rotation.from_euler('xyz',action[3:6]).as_matrix()
            local[:3,3] = action[:3]
            desired = self.previous_command@local
            R = rotation(frame.raw_state21)
            action = np.r_[desired[:3,3]-frame.ee_position,
                           Rotation.from_matrix(R.T@desired[:3,:3]).as_euler('xyz'),-action[6]].astype(np.float32)
        # Track the same clamped requested pose as the shared Cartesian controller.
        requested = absolute20_from_canonical(frame.ee_position,frame.raw_state21[11:17],action)
        self.previous_command = poses(requested[None])[0]
        self._pending_frame = frame
        return action

    def record_execution(self, info):
        """Call with BodyGraspEnvironment.step() info, including safety clamping."""
        from VLA.new_vlas.action_contracts import absolute20_from_canonical
        from VLA.admission_r100.seer_data import poses
        if self._pending_frame is None:
            raise ValueError('No pending requested action')
        action=np.asarray(info['executed_canonical_action'],np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all():
            raise ValueError('Actual executed canonical7 is required')
        f=self._pending_frame
        requested=absolute20_from_canonical(f.ee_position,f.raw_state21[11:17],action)
        self.previous_command=poses(requested[None])[0]
        self._pending_frame=None
