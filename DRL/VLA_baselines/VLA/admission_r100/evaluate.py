"""R97 durable physical loop, native NumPy1 map reference, optional real RGB-D."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import base64,hashlib,json,zlib
import numpy as np
import requests
from .common import read,TASKS
from VLA.postfit800_r98.common import durable_json

FINAL=list(range(20310001,20310201))
DEV=list(range(20311001,20311011));CONF=list(range(20312001,20312011));SMOKE=[20313001]


def approved(method,cohort):
    return bool(cohort['role']=='fresh_secondary_final800' and cohort['seeds']==FINAL
        and method.get('family') in ('seer','falcon_esm','gr00t_n15')
        and method.get('R100_frozen_before_final') is True
        and method.get('frozen_for_user_requested_benchmark') is True)


@contextmanager
def callbacks():
    import VLA.autorefinement_r97.evaluate as evaluator
    import VLA.autorefinement_r97.audit as auditor
    import VLA.autonomy_r83.evaluate as physical
    import VLA.receipt_recovery_r88.evaluate as bookkeeping
    import VLA.receipt_recovery_r88.map_reference as reference
    import VLA.recovery_r85.evaluate as writer
    saved=(evaluator.approved_final,auditor.approved_final,physical.render,bookkeeping.render,writer.atomic_json)
    evaluator.approved_final=auditor.approved_final=approved
    physical.render=bookkeeping.render=reference.render
    writer.atomic_json=durable_json
    try:yield
    finally:evaluator.approved_final,auditor.approved_final,physical.render,bookkeeping.render,writer.atomic_json=saved


def audit_result(path,require_complete=True):
    from VLA.autorefinement_r97.audit import audit
    with callbacks():return audit(path,require_complete=require_complete)


@contextmanager
def depth_transport(enabled):
    if not enabled:yield;return
    import VLA.autonomy_r83.evaluate as physical
    from VLA.falcon_esm_r91.depth import capture
    from VLA.scene64_conditioning import model_input
    original_factory=physical.environment_class_for_profile;session=requests.Session;latest={}
    def factory(profile):
        base=original_factory(profile)
        class Paired(base):
            def __init__(self,*args,**kwargs):
                self._r100_depth_ready=False;super().__init__(*args,**kwargs);self._r100_depth_ready=True
            def paired(self,obs):
                if not self._r100_depth_ready:return
                depth,_,_,K,cv=capture(self,obs)
                latest.clear();latest.update(raw=model_input(self,obs)['scene64'],depth=depth,K=K,cv=cv)
            def reset(self,*args,**kwargs):
                obs=super().reset(*args,**kwargs);self.paired(obs);return obs
            def step(self,action):
                result=super().step(action);self.paired(result[0]);return result
        return Paired
    class PairedSession(session):
        def post(self,url,**kwargs):
            payload=kwargs['json']
            if not np.array_equal(np.asarray(payload['scene64'],np.float32),latest['raw']):
                raise ValueError('RGB-D capture does not belong to the queried current observation')
            payload.update(front_depth_zlib_f32=base64.b64encode(zlib.compress(latest['depth'].astype('<f4').tobytes(),1)).decode(),
                front_intrinsics=latest['K'].tolist(),front_extrinsics_OpenCV=latest['cv'].tolist())
            return super().post(url,**kwargs)
    physical.environment_class_for_profile=factory;requests.Session=PairedSession
    try:yield
    finally:physical.environment_class_for_profile=original_factory;requests.Session=session


def main(a):
    from VLA.postfit800_r98.evaluate import check_pending_journal
    from VLA.autorefinement_r97.evaluate import main as actual_loop
    if a.resume_progress:check_pending_journal(a.resume_progress,read(a.cohort))
    with callbacks(),depth_transport(read(a.method)['family']=='falcon_esm'):actual_loop(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('method','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--resume-progress',type=Path)
    p.add_argument('--memory-fraction',type=float,default=.86);main(p.parse_args())
