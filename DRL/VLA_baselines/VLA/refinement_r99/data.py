"""Task/target-preserving TRAIN rehearsal, no observation/action changes."""
from collections import defaultdict, Counter
import numpy as np

from VLA.recovery_quality_r90.data import Data as ParentData


def regimes(raw, action):
    raw, action = np.asarray(raw),np.asarray(action)
    if raw.ndim!=2 or raw.shape[1]!=64 or action.shape!=(len(raw),7):
        raise ValueError('Measured TRAIN64 and aligned canonical7 required')
    held=raw[:,17]>.5
    unheld=raw[:,31]<.5
    contact=np.any(raw[:,28:30]>.5,axis=1)
    lateral=np.linalg.norm(action[:,:2],axis=1)>.003
    return {
        'safe_lateral_approach':unheld & lateral & (action[:,2]>=-.002),
        'contact_or_grip_transition':unheld & (contact | ((raw[:,10]>.5)!=(action[:,6]>0))),
        'held_transport':held & (action[:,6]>0) & (np.linalg.norm(action[:,:3],axis=1)>.003),
        'full_support':np.ones(len(raw),bool),
    }


class Data(ParentData):
    def __init__(self):
        super().__init__()
        self.rehearsal=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        self.stage_exposure=Counter()
        for ei,row in enumerate(self.rows):
            n=self.numeric(ei)
            for name,mask in regimes(n['scene64'],n['canonical7']).items():
                ids=np.flatnonzero(mask)
                if len(ids):self.rehearsal[row['task']][row['pick_name']][name].append((ei,ids))

    def draw(self,rng):
        if rng.random()<.5:
            result=super().draw(rng)
            self.stage_exposure['unchanged_R90']+=1
            return result
        task=str(rng.choice(sorted(self.groups)))
        target=str(rng.choice(sorted(self.groups[task]['complete_original'])))
        modes=self.rehearsal[task][target]
        names=['safe_lateral_approach','full_support'] if task=='touch' else [
            'safe_lateral_approach','contact_or_grip_transition','held_transport','full_support']
        name=str(rng.choice(names))
        actual=name if modes.get(name) else 'full_support'
        options=modes[actual]
        ei,ids=options[int(rng.integers(len(options)))]
        self.exposure[self.rows[ei]['source_kind']]+=1
        self.stage_exposure[task+'/'+actual]+=1
        return int(ei),int(rng.choice(ids))
