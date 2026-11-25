"""Read-only cross-check of completed R59 trials; never a tuning callback."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.observation_profile import profile_hash
from VLA.quantitative_r59.common import ROOT, OUT, CP, TASKS, SEEDS, read


def main():
    plan=read(OUT/'PREDECLARATION.json')
    protocol=read(plan['protocol']);profile=read(protocol['observation_profile_file'])
    checks={'protocol_file_unchanged':sha256_file(plan['protocol'])==plan['protocol_sha256'],
        'observation_profile_unchanged':profile_hash(profile)==protocol['observation_profile_sha256'],
        'checkpoint_master_unchanged':sha256_file(CP/'trainable_masters.pt')==plan['checkpoint_master_sha256'],
        'checkpoint_config_unchanged':sha256_file(CP/'config.json')==plan['checkpoint_config_sha256'],
        'schedule_unchanged':sha256_file(plan['schedule'])==plan['schedule_sha256']}
    checks.update({path:sha256_file(ROOT/path)==digest
        for path,digest in plan['environment_fingerprint']['sources_sha256'].items()})
    if not all(checks.values()):raise ValueError('Frozen plant/protocol/checkpoint changed')
    p=read(OUT/'episodes/progress.json');rows=p['rows']
    keys=[(r['seed'],r['task']) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<={(s,t) for s in SEEDS for t in TASKS}:
        raise ValueError('Duplicate or unplanned episode')
    receipts={}
    for line in (OUT/'episodes.neural_receipts.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue  # An in-flight append is not a completed-trial error.
        receipts[r['input_sha256']]=r
    counts={t:Counter() for t in TASKS};full=partial=0;errors=[];cache={}
    for row in rows:
        path=row['trajectory']
        if path not in cache:cache[path]=read(path)
        trace=cache[path][row['task']]
        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=row['success']:
            errors.append({'seed':row['seed'],'task':row['task'],'error':'summary/trace outcome mismatch'})
        c=counts[row['task']];c['trials']+=1;c['successes']+=bool(row['success'])
        c['safety_aborts']+=bool(row.get('safety_violation'))
        c['ordinary_failures']+=not row['success'] and not row.get('safety_violation')
        prior_hold=any(r[phase]['correct_object_held'] for r in trace for phase in ('before','after'))
        final_hold=trace[-1]['after']['correct_object_held']
        if row['success']:
            c['success_with_verified_attachment_at_terminal']+=bool(final_hold)
            c['success_with_any_prior_verified_attachment']+=bool(prior_hold)
            c['success_without_any_verified_attachment']+=not prior_hold
        chunks={}
        for r in trace:chunks.setdefault(r['chunk_query_step'],[]).append(r)
        for chunk in chunks.values():
            key=chunk[0]['neural_chunk_input_sha256'];receipt=receipts.get(key)
            if receipt is None or receipt['actual_native_visual_receipt'].get('actual_views')!=3:
                errors.append({'seed':row['seed'],'task':row['task'],'error':'actual3view neural receipt absent'});continue
            native=np.asarray([r['native_neural_action'] for r in chunk],np.float32)
            if len(chunk)==30:
                if hashlib.sha256(native.tobytes()).hexdigest()!=receipt['action_sha256_float32']:
                    errors.append({'seed':row['seed'],'task':row['task'],'error':'native chunk receipt mismatch'})
                full+=1
            else:partial+=1
    result={'passed_for_completed_prefix':not errors,'not_a_final800_certificate':len(rows)!=800,
        'snapshot_UTC':datetime.now(timezone.utc).isoformat(),'completed_trials_checked':len(rows),
        'frozen_plant_camera_protocol_checkpoint_checks':checks,'full_native30_chunks_hash_checked':full,
        'partial_terminal_chunks_not_claimed_full_hash_verified':partial,'per_action':counts,
        'all_safety_aborts_remain_in_primary_denominator':True,
        'secondary_hold_counts_do_not_change_frozen_success':True,
        'touch_does_not_require_grasp_attachment':True,
        'move_put_primary_does_not_require_prior_hold_release_or_settling':True,
        'no_video_files':not any((OUT/'episodes').rglob('*.mp4')),'errors':errors}
    atomic_json(OUT/'INDEPENDENT_COMPLETED_PREFIX_AUDIT.json',result)
    print(json.dumps({'checked':len(rows),'passed':not errors,'full_chunk_hashes':full,'counts':counts},ensure_ascii=False))
    if errors:raise ValueError('Independent completed-prefix audit failed')


if __name__=='__main__':main()
