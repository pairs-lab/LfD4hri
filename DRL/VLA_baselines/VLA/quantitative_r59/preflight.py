"""Declare the whole cohort before any outcome; qualify geometry, not SR."""
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.observation_profile import profile_hash
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from .common import ROOT, OUT, CP, SCHEDULE, SEEDS, TASKS, read


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PREDECLARATION.json').exists():
        raise ValueError('Cohort already frozen; do not silently re-declare')
    review = ROOT/'VLA/qualitative/ur5e_xvla_put_r59_independent_review_20261017/REVIEW_COMPLETE.json'
    if not read(review)['passed']: raise ValueError('Qualitative independent review incomplete')
    saved = read(CP/'saved_values.json')
    for name, key in (('config.json','config_sha256'), ('trainable_masters.pt','trainable_master_sha256')):
        if sha256_file(CP/name) != saved[key]: raise ValueError('Frozen weights changed')
    schedule = read(SCHEDULE)
    if schedule['checkpoint'] != str(CP) or any(schedule['tasks'][t]['selected_prefix'] != 30 for t in TASKS):
        raise ValueError('Frozen native30 execution changed')
    from VLA.run_feedback_refinement_r52 import qualify_final_seeds
    geometry = qualify_final_seeds(SEEDS)
    # Compare against additional post-R52 development/test geometry, without
    # reading any policy outcomes to select these new scenes.
    old_keys = set()
    evidence = []
    sources = [ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/geometry_exclusion.json']
    for path in sources:
        value = read(path)
        def collect(item):
            if isinstance(item, dict):
                if 'physical_scene_sha256' in item: old_keys.add(item['physical_scene_sha256'])
                for v in item.values(): collect(v)
            elif isinstance(item, list):
                for v in item: collect(v)
        collect(value); evidence.append({'path': str(path), 'sha256': sha256_file(path)})
    overlap = old_keys & {r['physical_scene_sha256'] for r in geometry['cases']}
    if overlap: raise ValueError('Fresh geometry overlaps old final/development')
    protocol = read(PROFILE_PROTOCOL); profile = read(protocol['observation_profile_file'])
    if profile_hash(profile) != protocol['observation_profile_sha256']: raise ValueError('Physical/camera profile changed')
    atomic_json(OUT/'geometry_exclusion.json', {**geometry, 'additional_geometry_evidence': evidence,
        'additional_overlap': 0, 'outcome_based_seed_selection': False})
    atomic_json(OUT/'cohort.json', {'role':'untouched_final',
        'scope':'R59_FROZEN_800_NEW_ENVIRONMENT_SCENES_NO_TEST_TUNING_NO_VIDEO', 'seeds': SEEDS})
    atomic_json(OUT/'PREDECLARATION.json', {'checkpoint': str(CP),
        'checkpoint_master_sha256': saved['trainable_master_sha256'], 'checkpoint_config_sha256': saved['config_sha256'],
        'schedule': str(SCHEDULE), 'schedule_sha256': sha256_file(SCHEDULE),
        'seeds': SEEDS, 'tasks': list(TASKS), 'per_action':200, 'total':800,
        'same_seed_block_has_four_different_task_scenes': True,
        'native_horizon_and_executed_prefix':30, 'image_size':384, 'control_hz':10,
        'same_two_RGB_plus_actual_third_sim_GT_image_and_67_numeric_fields':True,
        'numeric_tail_normalized_clip5_preserved_NOT_unlimited_lossless_information':True,
        'neural_VLA_action_source_NOT_oracle':True, 'terminal_release_supervisor':False,
        'video_or_snapshot_saved':False, 'infrastructure_errors_NOT_task_failures':True,
        'protocol':str(PROFILE_PROTOCOL), 'protocol_sha256':sha256_file(PROFILE_PROTOCOL),
        'protocol_historical_text_superseded': ['old pilot count', 'old no-privileged-policy claim'],
        'actual_primary_success':protocol['success'], 'environment_fingerprint':environment_runtime_fingerprint(profile),
        'put_primary_is_legacy_goal_region_NOT_release_and_settle':True,
        'qualitative_review':str(review), 'qualitative_review_sha256':sha256_file(review),
        'statistical_plan': 'Per action Wilson95; overall scene-block bootstrap95. Four future cohort mean/std are explicitly predictions, not measured seeds or training-seed variance.',
        'run_changes': 'Only output video/snapshot disabled and trajectory writes buffered until task boundary; no dynamics or policy change.',
        'source_sha256':{str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}})
    print({'predeclared':800, 'TRAIN_validation_and_old_geometry_overlap':0}, flush=True)


from pathlib import Path
if __name__ == '__main__': main()
