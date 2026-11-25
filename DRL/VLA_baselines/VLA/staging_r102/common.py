"""Independent staging paths and numerical contracts; no policy writes."""
from pathlib import Path
import numpy as np
from VLA.admission_r100.common import ROOT, OUT as R100, read, utc, atomic_json

OUT = ROOT / 'VLA/audits/native_staging_r102_20261025'


def depth_contract(depth, indices, frames):
    if depth.shape != (frames,384,384) or not np.array_equal(indices,np.arange(frames)):
        raise ValueError('Depth payload shape/order differs from actual RGB/labels')
    if not np.isfinite(depth).all() or np.any(depth<=0):
        raise ValueError('Nonfinite/nonpositive source metric depth')
    valid=(depth>.01)&(depth<3.)
    minimum=int(valid.sum(axis=(1,2)).min())
    if minimum<100:raise ValueError('Native metric-depth mask effectively empty')
    return {'frames':frames,'min_valid_pixels':minimum,
        'min_depth_m':float(depth.min()),'max_depth_m':float(depth.max())}


def geometry_contract(K,cv):
    if K.shape!=(3,3) or cv.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(cv).all():
        raise ValueError('Nonfinite/malformed camera geometry')
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-6):
        raise ValueError('Wrong native intrinsic convention')
    if not np.allclose(cv[3],[0,0,0,1],atol=1e-6):raise ValueError('Nonhomogeneous OpenCV extrinsics')
    R=cv[:3,:3]
    if not np.allclose(R@R.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):
        raise ValueError('Invalid/reflected camera rotation')


def gpu_admissible(r):
    return bool(r['RAM_available_MiB']>=4608 and r['GPU_free_MiB']>=7168)


def stop_reason(r,current,heartbeat_age,GPU):
    if r['RAM_available_MiB']<1800 or r['GPU_free_MiB']<2560:
        return 'resource headroom; preserve current trainer'
    if GPU and current.get('family')!='gr00t_n15':
        return 'primary scheduler changed family; preserve its next GPU allocation'
    if GPU and current.get('stage') not in ('adapt2048_attempt00','adapt4096_attempt00'):
        return 'primary phase changed; reserve native fresh reload/evaluation admission'
    if GPU and heartbeat_age>90:return 'primary native update heartbeat exceeded90s'
    return None
