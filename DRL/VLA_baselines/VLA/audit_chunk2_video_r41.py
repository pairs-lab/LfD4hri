"""Use the existing exact post-action video decoder on a local metadata snapshot.

The live development progress lacks the generic decoder's model/checkpoint
headers. Add them ONLY to an explicit audit snapshot, never to the live run.
All four outcomes of the predeclared first seed are included. No rerender.
"""
import argparse
import json
from pathlib import Path
import sys
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA import audit_r33_neural_video as decoder


def main(root,out):
    root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'plan.json').read_text());progress=json.loads((root/'progress.json').read_text())
    seed=plan['seeds'][0];rows=[r for r in progress['rows'] if r['seed']==seed]
    if tuple(r['task'] for r in rows)!=decoder.TASKS:raise ValueError('Fixed first seed all four terminal tasks required')
    for row in rows:
        traces=json.loads(Path(row['trajectory']).read_text())[row['task']]
        if any(not f['native_queue_audit']['queued_value_and_private_noise_verified'] for f in traces):
            raise ValueError('Actual native action queue proof missing')
    snapshot=out/'fixed_first_seed_metadata_snapshot.json'
    atomic_json(snapshot,{'model':'SmolVLA_native10_official_queue2_DEVELOPMENT',
        'checkpoint':plan['checkpoint'],'rows':rows,'original_progress':str(root/'progress.json'),
        'progress_IS_live_only_terminal_first_seed_is_snapshotted':True,'does_NOT_overwrite_live_progress':True})
    old=sys.argv
    try:
        sys.argv=[old[0],'--progress',str(snapshot),'--seed',str(seed),'--output',str(out/'decoded')]
        decoder.main()
    finally:sys.argv=old
    atomic_json(out/'audit.json',{'passed':True,'scope':'exact_first_seed_actual_video_and_native_queue_NO_new_SR',
        'all_four_tasks_including_failures':True,'source_plan_sha256':sha256_file(root/'plan.json'),
        'source_progress_NOT_modified':True,'snapshot_sha256':sha256_file(snapshot),
        'decoded_audit':str(out/'decoded/audit.json'),'implementation_sha256':sha256_file(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
