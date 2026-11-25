"""Same original images/labels/state, plus a causal simulator diagram."""
from VLA.new_vlas.dataset import NativeData as OriginalNativeData
from .scene_map import render


class MapData(OriginalNativeData):
    def sample(self, episode, frame, family):
        sample = super().sample(episode, frame, family)
        sample['privileged_map'] = render(self.numeric(episode)['scene64'][frame],sample['task'])
        return sample
