"""All-frame camera validation after exhaustive payload hashing, no GPU."""
from pathlib import Path
import numpy as np
from .common import OUT,R100,read,utc,atomic_json
from VLA.common.provenance import sha256_file


def camera_histories(views,projection,frames):
    if views.size!=16*frames or projection.size not in (16,16*frames):
        raise ValueError('Wrong camera/projection history length')
    V=np.asarray(views,np.float64).reshape(frames,4,4).swapaxes(-1,-2)
    P=np.asarray(projection,np.float64).reshape(-1,4,4).swapaxes(-1,-2)
    if not np.isfinite(V).all() or not np.isfinite(P).all():raise ValueError('Nonfinite camera history')
    if not np.allclose(V[:,3,:],[0,0,0,1],atol=1e-6):raise ValueError('Nonhomogeneous view history')
    rotations=V[:,:3,:3]
    if not np.allclose(rotations@rotations.swapaxes(-1,-2),np.eye(3),atol=1e-5):
        raise ValueError('Nonorthogonal camera in middle of trajectory')
    if not np.allclose(np.linalg.det(rotations),1,atol=1e-5):raise ValueError('Reflected camera history')
    if np.any(P[:,0,0]<=0) or np.any(P[:,1,1]<=0):raise ValueError('Nonpositive camera focal length')
    if not np.allclose(P[:,3,2],-1,atol=1e-6) or not np.allclose(P[:,3,3],0,atol=1e-6):
        raise ValueError('Wrong OpenGL perspective projection convention')
    return {'frames':frames,'projection_count':len(P),
        'fx_range':[float((192*P[:,0,0]).min()),float((192*P[:,0,0]).max())]}


def main():
    proof=read(OUT/'FULL_DEPTH_SCAN.json')
    index_path=R100/'falcon_esm/DEPTH_INDEX.json'
    if not proof['passed'] or sha256_file(index_path)!=proof['index_sha256']:
        raise ValueError('Require unchanged fully hashed depth scan')
    frames=0
    for ei,row in enumerate(read(index_path)):
        previous=read(OUT/'depth_rows'/f'row{ei:04d}.json');stat=Path(row['file']).stat()
        if (previous['payload_sha256']!=row['sha256'] or previous['payload_stat']!=
            {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}):raise ValueError('Hashed payload changed')
        with np.load(row['file'],allow_pickle=False) as values:
            camera_histories(values['view_world_to_OpenGL_camera'],values['projection_OpenGL'],row['frames'])
        frames+=row['frames']
    if frames!=proof['frames']:raise ValueError('Missing camera frames')
    atomic_json(OUT/'ALL_FRAME_GEOMETRY.json',{'passed':True,'UTC':utc(),'frames':frames,
        'every_view_and_projection_checked_not_only_reset_terminal':True,
        'payload_hashes_reused_from_FULL_DEPTH_SCAN_with_unchanged_size_mtime':True,
        'actual_neural_episodes':0,'visibility_NOT_proven_by_camera_matrix_validation':True})


if __name__=='__main__':main()
