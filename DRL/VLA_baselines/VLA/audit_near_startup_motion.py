"""Read-only control instrumentation of the exact stopped TRAIN startup.

No altered proposals, thresholds, controls, objects or simulator guards.
Records every candidate, including those that fail before camera validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.common.physics_safety import surface_clearance
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.near_object_recovery_start import closed_empty_near_object_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan_path = args.failed_candidate / 'collection_plan.json'
    state_path = args.failed_candidate / 'collection_state.json'
    plan, state = json.loads(plan_path.read_text()), json.loads(state_path.read_text())
    if state['status'] != 'stopped_error_NOT_training' or state['completed_episodes'] % 2:
        raise ValueError('Requires exact stopped complete-pair TRAIN boundary')
    source = Path(plan['source'])
    filename = plan['selected_source_files'][state['completed_episodes'] // 2]
    row = next(r for r in json.loads((source / 'index.json').read_text()) if r['file'] == filename)
    if json.loads((source / 'dataset_manifest.json').read_text()).get('split') != 'train':
        raise ValueError('TRAIN only')
    files = [plan_path, state_path, source / 'index.json', source / 'dataset_manifest.json',
             source / filename, Path(__file__), Path(__file__).with_name('near_object_recovery_start.py')]
    frozen = {str(p.resolve()): sha256_file(p) for p in files}
    args.output.mkdir(parents=True)
    traces = []
    exception = None
    with audit_environment(source, image_size=384, required_success_steps=1,
                           use_semantic_success=False, render_observations=False) as env:
        reset_source_episode(env, source, row)
        original_reset, original_step = env.reset, env.step
        def reset(task):
            observation = original_reset(task)
            traces.append({'initial_tcp': np.asarray(env.gripper.grasp_center()).tolist(), 'steps': []})
            return observation
        def step(action):
            before_tcp = np.asarray(env.gripper.grasp_center()).copy()
            before_clearance = float(surface_clearance(env))
            result = original_step(action)
            traces[-1]['steps'].append({'before_tcp': before_tcp.tolist(),
                'after_tcp': np.asarray(env.gripper.grasp_center()).tolist(),
                'clearance_before_m': before_clearance,
                'clearance_after_m': float(surface_clearance(env)),
                'requested_action': np.asarray(action).tolist(),
                'executed_action': result[3]['executed_canonical_action'],
                'safety_clamped': result[3]['safety_clamped_action'],
                'done': bool(result[2]), 'safety_violation': result[3].get('safety_violation')})
            if len(traces[-1]['steps']) % 10 == 0:
                (args.output / 'motion_progress.json').write_text(json.dumps(traces, indent=2) + '\n')
            return result
        env.reset, env.step = reset, step
        try:
            closed_empty_near_object_start(env, TaskSpec.from_dict(row['task']))
        except RuntimeError as exc:
            exception = str(exc)
        for camera in ('front', 'wrist'):
            Image.fromarray(env.render(camera)).save(args.output / f'last_candidate_{camera}.png')
    report = {'source_episode': filename, 'source': str(source), 'exception': exception,
              'original_rejection_message_identical': exception == state['error'],
              'all_candidate_motion_traces': traces, 'frozen_files': frozen,
              'training_performed': False, 'final_test_used': False,
              'controls_and_guards_changed': False}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    if any(sha256_file(Path(p)) != h for p, h in frozen.items()):
        raise RuntimeError('Source or code changed during motion diagnosis')
    print(json.dumps({'original_rejection_message_identical': report['original_rejection_message_identical'],
                      'candidates': len(traces), 'controls': [len(t['steps']) for t in traces]}))


if __name__ == '__main__':
    main()
