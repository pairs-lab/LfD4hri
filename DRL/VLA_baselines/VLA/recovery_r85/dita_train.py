"""Fix four-action gate sampling, not its acceptance/gradient checks or native model."""
import argparse
from pathlib import Path
from .common import bindings

def balanced_gate(data,index):
    from VLA.family_transfer_r54.data import anchors
    ids=anchors(data)
    return ids[index%len(ids)]

def main(a):
    import VLA.dita_transfer_r84.train as old
    old_bindings=old.bindings
    old.bindings=lambda:{**old_bindings(),**bindings()}
    if a.op=='reload':return old.reload(a)
    if not a.four_task_gate:return old.fit(a)
    if a.updates!=4 or a.accumulation!=4 or a.device!='cuda' or a.resume is None:
        raise ValueError('Only the declared2-CPU +2-CUDA optimizer gate may use this anchor sampler')
    from VLA.replacement_r84 import data as shared
    original=shared.Data
    class GateData(original):
        def __init__(self):
            super().__init__();self.gate_index=0
        def draw(self,rng):
            pair=balanced_gate(self,self.gate_index);self.gate_index+=1
            return pair
    shared.Data=GateData
    try:old.fit(a)
    finally:shared.Data=original

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='op',required=True)
    f=sub.add_parser('fit');f.add_argument('--output',type=Path,required=True);f.add_argument('--updates',type=int,required=True)
    f.add_argument('--accumulation',type=int,default=32);f.add_argument('--resume',type=Path)
    f.add_argument('--device',choices=('cpu','cuda'),default='cuda');f.add_argument('--four-task-gate',action='store_true')
    r=sub.add_parser('reload');r.add_argument('--checkpoint',type=Path,required=True)
    r.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    main(p.parse_args())
