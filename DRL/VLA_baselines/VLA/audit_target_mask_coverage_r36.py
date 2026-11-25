"""Check BOTH causal target-mask streams in every unchanged fitting episode."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def summarize(front,wrist,steps):
    if (front.dtype!=np.bool_ or wrist.dtype!=np.bool_ or front.shape!=(steps,384,384)
            or wrist.shape!=front.shape):raise ValueError('Complete aligned current masks bool[N,384,384] required')
    f=np.count_nonzero(front,axis=(1,2));w=np.count_nonzero(wrist,axis=(1,2));best=np.maximum(f,w)
    return {'frames':steps,'chosen_front':int(np.count_nonzero(f>=w)),'chosen_wrist':int(np.count_nonzero(w>f)),
        'both_current_masks_empty':int(np.count_nonzero(best==0)),
        'best_view_at_least48_raw_pixels':int(np.count_nonzero(best>=48)),
        'min_best_visible_pixels':int(best.min()),'max_best_visible_pixels':int(best.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('projection','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    projection=json.loads((a.projection/'audit.json').read_text());source=Path(projection['source'])
    admission=json.loads(a.admission.read_text());correction=Path(admission['candidate'])
    qualified={r['file'] for r in admission['rows']}
    groups=[('original',source,json.loads((source/'index.json').read_text())),
        ('canonical_correction',correction,[r for r in json.loads((correction/'index.json').read_text()) if r.get('file') in qualified])]
    if len(groups[0][2])!=2107 or len(groups[1][2])!=38:raise ValueError('Full unchanged2107+38 source required')
    results=[];counts=Counter();by_task={t:Counter() for t in ('touch','pick','move','pick_place')}
    for kind,root,rows in groups:
        for row in rows:
            with np.load(root/row['file'],allow_pickle=False) as z:
                info=summarize(z['front_pick_mask'],z['wrist_pick_mask'],row['steps'])
            result={'kind':kind,'file':row['file'],'task':row['task']['name'],'target':row['task']['pick_name'],**info};results.append(result)
            for key in ('frames','chosen_front','chosen_wrist','both_current_masks_empty','best_view_at_least48_raw_pixels'):
                counts[key]+=info[key];by_task[row['task']['name']][key]+=info[key]
            if len(results)%40==0:atomic_json(a.output/'progress.json',{'completed_episodes':len(results),'planned_episodes':2145,'frames':counts['frames']})
    if counts['frames']!=117736:raise ValueError('Actual full frame coverage changed')
    atomic_json(a.output/'audit.json',{'passed':True,'scope':'all_actual_original_and_canonical_correction_target_masks_NOT_model_SR',
        'episodes':len(results),'counts':dict(counts),'by_task':{k:dict(v) for k,v in by_task.items()},'rows':results,
        'occluded_objects_synthetically_rerendered':False,'original_images_actions_or_split_changed':False,
        'both_empty_behavior':'use current full-front image; no hallucinated hidden target',
        'bound_files':{str(path.resolve()):sha256_file(path) for path in (Path(__file__),a.projection/'audit.json',a.admission,source/'index.json',correction/'index.json')},
        'training_qualified':False,'model_SR_established':False})


if __name__=='__main__':main()
