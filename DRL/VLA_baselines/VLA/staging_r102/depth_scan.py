"""Full TRAIN payload scan, bounded-memory/resumable; no neural policy calls."""
import gc,os,time
from pathlib import Path
from collections import Counter
import numpy as np
from .common import OUT,R100,read,utc,atomic_json,depth_contract,geometry_contract
from VLA.common.provenance import sha256_file
from VLA.falcon_esm_r91.data import geometry


def main():
    os.nice(10);OUT.mkdir(parents=True,exist_ok=True)
    proof=read(R100/'falcon_esm/DEPTH_ADMISSION.json')
    index_path=R100/'falcon_esm/DEPTH_INDEX.json'
    digest=sha256_file(index_path)
    if not proof['passed'] or digest!=proof['depth_index_sha256']:raise ValueError('Frozen index changed')
    rows=read(index_path)
    if len(rows)!=proof['episodes'] or sum(r['frames'] for r in rows)!=proof['frames']:
        raise ValueError('Incomplete depth index')
    from VLA.refinement_r99.data import Data
    data=Data()
    if len(data.rows)!=len(rows):raise ValueError('Depth/TRAIN row count differs')
    destination=OUT/'depth_rows';destination.mkdir(exist_ok=True)
    totals=Counter();tasks=Counter();targets=set();started=time.monotonic()
    for ei,(row,source) in enumerate(zip(rows,data.rows,strict=True)):
        if row['source_sha256']!=source['source_sha256'] or row['frames']!=source['frames']:
            raise ValueError('Depth/RGB/source chronology differs')
        record=destination/f'row{ei:04d}.json'
        stat=Path(row['file']).stat()
        layout={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if record.exists():
            old=read(record)
            if old['index_sha256']!=digest or old['payload_sha256']!=row['sha256'] or old['payload_stat']!=layout:
                raise ValueError('Cannot reuse a receipt from another frozen payload/index')
        else:
            # Includes all2185 original payloads, not merely their old pairing receipts.
            if sha256_file(row['file'])!=row['sha256']:raise ValueError('Corrupt TRAIN depth: '+row['file'])
            with np.load(row['file'],allow_pickle=False) as values:
                depth=values['front_depth_z_m'];frames=row['frames']
                metrics=depth_contract(depth,values['source_frame_indices'],frames)
                views=values['view_world_to_OpenGL_camera'];projection=values['projection_OpenGL']
                if len(views)!=frames:raise ValueError('Wrong view chronology')
                for fi in sorted({0,frames//2,frames-1}):
                    p=projection[fi] if projection.ndim>1 else projection
                    K,cv=geometry(views[fi],p);geometry_contract(K,cv)
                del depth
            old={'passed':True,'UTC':utc(),'index_sha256':digest,'payload_sha256':row['sha256'],
                'payload':row['file'],'payload_stat':layout,'row':ei,'task':source['task'],'target':source['pick_name'],
                'source_kind':source['source_kind'],**metrics}
            atomic_json(record,old)
        totals['rows']+=1;totals['frames']+=old['frames'];tasks[old['task']]+=old['frames']
        targets.add((old['task'],old['target']))
        if ei%10==0 or ei==len(rows)-1:
            atomic_json(OUT/'DEPTH_SCAN_STATUS.json',{'UTC':utc(),'status':'scanning_actual_TRAIN_depth',
                'rows':totals['rows'],'planned_rows':len(rows),'frames':totals['frames'],
                'per_action_frames':dict(tasks),'task_target_strata':len(targets),
                'elapsed_s':time.monotonic()-started,'actual_neural_episodes':0})
            print({'rows':totals['rows'],'frames':totals['frames']},flush=True);gc.collect()
    if len(targets)!=76:raise ValueError('Incomplete four-action/19-target TRAIN coverage')
    atomic_json(OUT/'FULL_DEPTH_SCAN.json',{'passed':True,'UTC':utc(),'rows':len(rows),
        'frames':totals['frames'],'per_action_frames':dict(tasks),'task_target_strata':len(targets),
        'all_indexed_payloads_hashed_and_decoded':True,'index_sha256':digest,
        'prior_physics_pairing_proofs_reused_NOT_replayed_here':True,
        'object_visibility_or_high_SR_NOT_proven':True,'actual_neural_episodes':0})
    atomic_json(OUT/'DEPTH_SCAN_STATUS.json',{'UTC':utc(),'status':'completed',**read(OUT/'FULL_DEPTH_SCAN.json')})


if __name__=='__main__':main()
