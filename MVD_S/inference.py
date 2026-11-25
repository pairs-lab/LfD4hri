"""MVD-S video inference with the same task-string interface as TSM."""
from pathlib import Path
import time
import numpy as np
import torch
from .checkpoint import read_checkpoint, load_classifier, sha256
from .preprocess import decode_samples, make_views

DEFAULT_LABEL_MAP = Path(__file__).with_name('labels.txt')


def load_labels(path=DEFAULT_LABEL_MAP):
    labels = Path(path).read_text(encoding='utf-8').splitlines()
    if len(labels) != 174 or any(not x.strip() for x in labels):
        raise ValueError('Label map must have exactly 174 nonempty lines in SSV2 class-index order')
    return [x.strip() for x in labels]


class Recognizer:
    def __init__(self, checkpoint, *, device='cpu', label_map=DEFAULT_LABEL_MAP,
                 expected_sha256=None, allow_smoke_checkpoint=False):
        self.labels = load_labels(label_map)
        self.label_map_sha256 = sha256(label_map)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; select --device cpu')
        state,self.checkpoint = read_checkpoint(checkpoint, expected_sha256)
        metadata = self.checkpoint.get('metadata',{})
        if metadata.get('software_smoke_only') and not allow_smoke_checkpoint:
            raise ValueError('Software-smoke checkpoint is not a trained action classifier')
        if metadata.get('label_map_sha256') and metadata['label_map_sha256'] != self.label_map_sha256:
            raise ValueError('Checkpoint/label-map mismatch')
        from .model import build_model
        self.model = build_model()
        load_classifier(self.model,state)
        del state
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, video, *, preset='paper', decoder='decord', batch_size=1, topk=5):
        if preset not in ('paper','fast') or batch_size < 1 or not 1 <= topk <= 174:
            raise ValueError('Invalid preset/batch size/topk')
        start = time.perf_counter()
        samples,metadata = decode_samples(video,clips=2 if preset=='paper' else 1,decoder=decoder)
        views,records = make_views(samples,crops=3 if preset=='paper' else 1)
        probabilities=[]
        for begin in range(0,len(views),batch_size):
            logits=self.model(torch.stack(views[begin:begin+batch_size]).to(self.device))
            if logits.shape[-1] != 174 or not torch.isfinite(logits).all():
                raise ValueError('Invalid native classifier output')
            probabilities.append(logits.float().softmax(-1).cpu())
        scores=torch.cat(probabilities).mean(0).numpy()  # Upstream merge: mean of per-view softmax.
        order=np.argsort(-scores,kind='stable')[:topk]
        return {'backend':'mvd_s','action':self.labels[int(order[0])],'class_index':int(order[0]),
                'topk':[{'class_index':int(i),'action':self.labels[int(i)],'score':float(scores[i])} for i in order],
                'scores':scores.tolist(),'checkpoint':self.checkpoint,
                'label_map_sha256':self.label_map_sha256,'device':str(self.device),
                'preset':preset,'num_views':len(views),'view_metadata':records,
                'video_metadata':metadata,'elapsed_seconds':time.perf_counter()-start,
                'software_smoke_only':bool(self.checkpoint.get('metadata',{}).get('software_smoke_only'))}
