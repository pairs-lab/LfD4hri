"""Record the diagnostic-only amendment BEFORE the first new neural episode."""
from datetime import datetime,timezone
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,CP,read


def main():
    if (OUT/'episodes').exists() or (OUT/'query_latency.jsonl').exists():
        raise ValueError('No retrospective source amendment after evaluating new episodes')
    if not read(OUT/'NO_VIDEO_NONINTERFERENCE.json')['passed']:
        raise ValueError('Actual used-input noninterference qualification required')
    data=Path(read(CP/'config.json')['data_root'])
    index=read(data/'index.json')
    groups={row['scene_group'] for row in index if row.get('scene_group')}
    missing=[row for row in index if not row.get('scene_group')]
    source_indices={}
    from VLA.audit_training_readiness import physical_scene_key
    for row in missing:
        path=Path(row['source']).parent/'index.json'
        if path not in source_indices:source_indices[path]=read(path)
        candidates=[r for r in source_indices[path] if
            r.get('file',r.get('filename'))==Path(row['source']).name]
        if len(candidates)!=1:
            raise ValueError('Cannot bind current TRAIN scene without geometry: '+row['source'])
        groups.add(physical_scene_key(candidates[0]['task']))
    new={row['physical_scene_sha256'] for row in read(OUT/'geometry_exclusion.json')['cases']}
    if groups & new:raise ValueError('Fresh800 geometry overlaps current TRAIN data')
    atomic_json(OUT/'current_TRAIN_geometry_exclusion.json',{'passed':True,
        'TRAIN_episodes':len(index),'unique_TRAIN_geometry_keys':len(groups),
        'TRAIN_index_sha256':sha256_file(data/'index.json'),
        'missing_group_rows_recovered_from_source_task':len(missing),'overlap':0,
        'source_index_sha256':{str(p):sha256_file(p) for p in source_indices},
        'no_neural_outcomes_used':True})
    plan=read(OUT/'PREDECLARATION.json');changes={}
    for path,original in plan['source_sha256'].items():
        actual=sha256_file(path)
        if actual!=original:
            if Path(path).name!='noninterference.py':
                raise ValueError('Only diagnostic instrumentation was amended, not rollout/model/scene selection')
            changes[path]={'original_sha256':original,'new_sha256':actual}
    atomic_json(OUT/'PRE_EPISODE_DIAGNOSTIC_AMENDMENT.json',{
        'UTC':datetime.now(timezone.utc).isoformat(), 'predeclaration_sha256':sha256_file(OUT/'PREDECLARATION.json'),
        'source_amendments':changes,'actual_new_evaluation_episodes_before_amendment':0,
        'reason':'Separate unused per-control RGB rasterization differences from actual chunk-boundary neural inputs. Preserve failed strict diagnostics, full step hashes and pixel differences. No model/action/environment/seed/threshold change.',
        'clarified_qualitative_review_sha256':sha256_file(Path(plan['qualitative_review'])),
        'qualitative_clarification':'Presentation camera is not policy wrist; actual scene11 policy front visibility7/7 and wrist absence queries explicitly reported.',
        'original_predeclaration_preserved':True})


if __name__=='__main__':main()
