"""Bounded post-exec registration with full argv, parent and start-time checks."""
import time
from pathlib import Path


def argv_equal(actual,requested):
    if not actual or actual[1:]!=requested[1:]: return False
    return actual[0]==requested[0] or Path(actual[0]).resolve()==Path(requested[0]).resolve()


def await_identity(child,requested,parent,read_identity,timeout=10,clock=time.monotonic,pause=time.sleep):
    deadline=clock()+timeout; first_start=None; previous=None; observations=[]
    while clock()<deadline:
        if child.poll() is not None:
            raise RuntimeError('Own child exited before full post-exec registration')
        current=read_identity(child.pid)
        if current:
            if first_start is None: first_start=current['start_ticks']
            if current['start_ticks']!=first_start or current['ppid']!=parent:
                raise ValueError('Child start-time or parent changed; refuse registration')
            if len(observations)<8 and (not observations or observations[-1]!=current):
                observations.append(current)
            if argv_equal(current['command'],requested):
                if current==previous:
                    return current,observations
                previous=current
            else:
                previous=None
        pause(.01)
    raise TimeoutError('Own process never matched full requested post-exec argv')
