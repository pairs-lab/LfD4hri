"""Physical-unit adapters only; no object/goal-dependent action synthesis."""
import numpy as np
from scipy.spatial.transform import Rotation
from VLA.new_vlas.action_contracts import rotation_matrix,absolute20_from_canonical
from .seer_data import poses


def measured_pose(position,columns):
    out=np.eye(4);out[:3,:3]=rotation_matrix(columns);out[:3,3]=position
    if not np.isfinite(out).all():raise ValueError('Nonfinite actual measured tool pose')
    return out


def decode_seer(native,current,previous_command):
    native=np.asarray(native,np.float64)
    if native.shape!=(3,7) or not np.isfinite(native).all():raise ValueError('Native Seer3x7 finite local chunk required')
    result=[];anchor=previous_command.copy();measured=current.copy();next_command=None
    for step in native:
        local=np.eye(4);local[:3,3]=step[:3];local[:3,:3]=Rotation.from_euler('xyz',step[3:6]).as_matrix()
        target=anchor@local
        delta=np.r_[target[:3,3]-measured[:3,3],Rotation.from_matrix(measured[:3,:3].T@target[:3,:3]).as_euler('xyz'),-step[6]]
        result.append(delta.astype(np.float32))
        columns=measured[:3,:2].T.reshape(6)
        bounded=absolute20_from_canonical(measured[:3,3],columns,delta)
        anchor=poses(bounded[None])[0]
        if next_command is None:next_command=anchor.copy()
        # Only first action is executed. Remaining native forecast is decoded
        # about its own preceding requested pose, never a future observation.
        measured=anchor.copy()
    return np.stack(result),next_command


def decode_depth(payload):
    import base64,zlib
    encoded=payload['front_depth_zlib_f32']
    if len(encoded)>2*1024*1024:raise ValueError('Oversized native depth transport')
    decompressor=zlib.decompressobj();expected=384*384*4
    raw=decompressor.decompress(base64.b64decode(encoded,validate=True),expected+1)
    if len(raw)!=expected or decompressor.unconsumed_tail or not decompressor.eof:
        raise ValueError('Malformed or oversized float32 depth payload')
    depth=np.frombuffer(raw,dtype='<f4').reshape(384,384)
    K=np.asarray(payload['front_intrinsics'],np.float32);cv=np.asarray(payload['front_extrinsics_OpenCV'],np.float32)
    if (not np.isfinite(depth).all() or np.min(depth)<=0 or K.shape!=(3,3) or cv.shape!=(4,4)
        or not np.isfinite(K).all() or not np.isfinite(cv).all()):raise ValueError('Invalid metric depth/calibration')
    return {'depth_z_m':depth,'intrinsics':K,'extrinsics_OpenCV':cv}
