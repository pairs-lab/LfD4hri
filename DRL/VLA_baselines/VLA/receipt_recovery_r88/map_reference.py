"""Audit reference for the original Dita NumPy1.26 renderer, not a policy image.

NumPy2 weak Python scalar promotion changes four ellipse pixels at the failing
scene. Explicit float64 scalar intermediates reproduce the SAME map consumed
by the unchanged NumPy1.26 Dita server. No guard is disabled or made tolerant.
"""
import inspect
from VLA.privileged_visual_r53 import scene_map


def build():
    source=inspect.getsource(scene_map.geometry)+'\n'+inspect.getsource(scene_map.render)
    changes={'raw[26]/2':'float(raw[26])/2',
             'radius*sx':'float(radius)*sx','radius*sy':'float(radius)*sy',
             'raw[24]*sx/2':'float(raw[24])*sx/2','raw[25]*sy/2':'float(raw[25])*sy/2'}
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('Map source changed; no broad unreviewed patch')
        source=source.replace(old,new)
    namespace={**scene_map.__dict__}
    exec(compile(source,'<R88 NumPy1.26-equivalent audit reference only>','exec'),namespace)
    return namespace['render']


render=build()
