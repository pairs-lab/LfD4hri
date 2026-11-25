"""Synchronized TRAIN depth + existing RGB/causal labels; no new final data."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from VLA.recovery_quality_r90.data import Data as RecoveryData
from VLA.family_transfer_r54.data import action_stats
from VLA.common.provenance import sha256_file
from .common import ROOT, OUT, read

DEPTH = ROOT/'VLA/audits/falcon_depth_pairing_r72_20261019/full_TRAIN_depth_r73'


def geometry(view, projection, width=384, height=384):
    """PyBullet column-major OpenGL -> metric OpenCV world-to-camera."""
    V = np.asarray(view,np.float64).reshape(4,4).T
    P = np.asarray(projection,np.float64).reshape(4,4).T
    cv = np.diag([1.,-1.,-1.,1.]) @ V
    K = np.array([[width*P[0,0]/2,0,(width-1)/2],
                  [0,height*P[1,1]/2,(height-1)/2],[0,0,1]],np.float64)
    if not np.isclose(np.linalg.det(cv[:3,:3]),1.,atol=1e-6):
        raise ValueError('Camera rotation is not a proper rigid transform')
    return K.astype(np.float32), cv.astype(np.float32)


class Data(RecoveryData):
    def __init__(self):
        super().__init__()
        admission = read(OUT/'DEPTH_ADMISSION.json')
        if not admission['passed'] or admission['episodes']!=len(self.rows):
            raise ValueError('All original/recovery RGB-D rows must be admitted first')
        self.depth_rows = read(OUT/'DEPTH_INDEX.json')
        if sha256_file(OUT/'DEPTH_INDEX.json')!=admission['depth_index_sha256']:
            raise ValueError('Depth bank index changed')
        self.horizon = 10

    @lru_cache(maxsize=2)
    def depth(self,ei):
        row = self.depth_rows[ei]
        if sha256_file(row['file']) != row['sha256']:
            raise ValueError('Paired depth changed')
        with np.load(row['file'],allow_pickle=False) as value:
            return {key:value[key] for key in ('front_depth_z_m','view_world_to_OpenGL_camera','projection_OpenGL')}

    def at(self,ei,fi):
        s = super().at(ei,fi)
        d = self.depth(ei)
        projection = d['projection_OpenGL']
        if projection.ndim>1: projection=projection[fi]
        K, cv = geometry(d['view_world_to_OpenGL_camera'][fi],projection)
        s.update(depth_z_m=d['front_depth_z_m'][fi].astype(np.float32),
            intrinsics=K, extrinsics_OpenCV=cv)
        s['action'][s['action_is_pad'],:6] = 0.
        return s


def statistics(data):
    # Same original TRAIN normalizers for appended privileged information.
    # Only action-statistics data are eligible TRAIN (including corrections).
    return {'scene67':data.stats['scene67'], 'actions':action_stats(data),
        'native_horizon':10,'numeric_input':'causal67','action':'world_delta_local_Euler_OPEN01',
        'native_normalized_pose_range':[-.65,.65]}
