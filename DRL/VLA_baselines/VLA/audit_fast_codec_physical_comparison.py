"""OFFLINE codec quality: matched raw/quantized labels, never VLA success.

The earlier terminal-only failure is retained. This measures both the frozen
first-hit evaluation predicate and terminal expert quality on those SAME four
predeclared cases. No model, fitting, changed score or scripted VLA fallback.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.collect_native_recovery_view import NativeCapture
from VLA.dataset_replay_context import audit_environment
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


def run(codec, output):
    proof = json.loads((codec / 'audit.json').read_text())
    if not proof['passed'] or proof['frames'] != 1479:
        raise ValueError('Complete actual codec evidence required')
    source, candidate = Path(proof['source']), Path(proof['candidate'])
    originals = {r['file']: r for r in json.loads((source / 'index.json').read_text())}
    index = json.loads((candidate / 'index.json').read_text())
    tasks = ('touch', 'pick', 'move', 'pick_place')
    selected = [next(r for r in index if r['task']['name'] == task) for task in tasks]
    protocol = json.loads(PROFILE_PROTOCOL.read_text())['success']
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for row in selected:
        entry = next(r for r in proof['rows'] if r['file'] == row['file'])
        if sha256_file(codec / row['file']) != entry['decoded_payload_sha256']:
            raise ValueError('Actual decoded labels changed')
        if sha256_file(candidate / row['file']) != row['payload_sha256']:
            raise ValueError('Actual native source changed')
        with np.load(candidate / row['file'], allow_pickle=False) as z:
            raw = z['canonical_actions'].copy()
            targets = z['next_joint_targets'].copy()
            closure = z['next_requested_gripper_closure'].copy()
        with np.load(codec / row['file'], allow_pickle=False) as z:
            quantized = z['canonical'].copy()
        if raw.shape != quantized.shape:
            raise ValueError('Do not trim either fixed sequence')
        for label, commands in (('unquantized_positive_control', raw), ('actual_FAST32_quantized', quantized)):
            env = audit_environment(source, image_size=384, control_hz=10, seed=0,
                required_success_steps=1, use_semantic_success=True,
                success_threshold_scale=protocol['reward_xml_threshold_scale'],
                semantic_success_thresholds=protocol['semantic_thresholds'])
            case = output / (row['task']['name'] + '_' + label)
            case.mkdir()
            result = {'task': row['task']['name'], 'file': row['file'], 'kind': label,
                'planned_controls': len(commands), 'error': None, 'frames': [],
                'first_semantic_success_step': None, 'first_strict_success_step': None,
                'first_safety_step': None, 'first_expert_quality_failure_step': None}
            try:
                capture = NativeCapture(env, source, originals[row['source_file']], row, targets, closure)
                obs = capture.reset(TaskSpec.from_dict(row['task']))
                for camera in ('front', 'wrist'):
                    Image.fromarray(getattr(obs, camera + '_rgb')).save(case / ('initial_' + camera + '.png'))
                wrapper = NearSurfaceTouchCollection(env, None, None) if row['task']['name'] == 'touch' else env
                for step, command in enumerate(commands, 1):
                    obs, _, _, info = wrapper.step(command)
                    values = {'step': step, 'strict': bool(info['strict_raw_success']),
                        'semantic': bool(info['semantic_raw_success']), 'safety': info.get('safety_violation'),
                        'expert_quality_failure': info.get('expert_quality_failure'),
                        'metrics': env.semantic_success_metrics(),
                        'correct_held': bool(env.gripper.grasped_object == env.pick_id),
                        'bilateral': bool(env.gripper._bilateral(env.pick_id))}
                    result['frames'].append(values)
                    for key, observed in (('first_semantic_success_step', values['semantic']),
                        ('first_strict_success_step', values['strict']),
                        ('first_safety_step', values['safety']),
                        ('first_expert_quality_failure_step', values['expert_quality_failure'])):
                        if observed and result[key] is None:
                            result[key] = step
                    if values['safety']:
                        break
                result['terminal_strict_success'] = bool(env.is_strict_success())
                result['terminal_semantic_success'] = bool(env.is_semantic_success())
                if row['task']['name'] == 'touch':
                    gap, force = wrapper.terminal_geometry()
                    result['terminal_gentle_touch_quality'] = bool(env.is_strict_success() and gap is not None and gap <= .0002 and force <= 5)
                    result.update(terminal_surface_gap_m=gap, terminal_max_pad_force_n=force)
                for camera in ('front', 'wrist'):
                    Image.fromarray(getattr(obs, camera + '_rgb')).save(case / ('terminal_' + camera + '.png'))
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
            finally:
                env.close()
            first, unsafe = result['first_semantic_success_step'], result['first_safety_step']
            result['offline_first_hit_before_safety'] = first is not None and (unsafe is None or first < unsafe)
            atomic_json(case / 'trace.json', result)
            reports.append({k: v for k, v in result.items() if k != 'frames'})
            atomic_json(output / 'progress.json', {'rows': reports, 'model_SR_established': False})
    positive = [r for r in reports if r['kind'] == 'unquantized_positive_control']
    atomic_json(output / 'audit.json', {'scope': 'matched_OFFLINE_codec_physical_quality_NOT_VLA_SR',
        'positive_control_all_terminal_strict': all(r.get('terminal_strict_success') and not r['error'] for r in positive),
        'rows': reports, 'codec_audit_sha256': sha256_file(codec / 'audit.json'),
        'implementation_sha256': sha256_file(Path(__file__)), 'protocol_sha256': sha256_file(PROFILE_PROTOCOL),
        'same_four_predeclared_cases_no_reroll': True, 'same_success_and_safety': True,
        'neural_model_executed': False, 'model_SR_established': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.codec.resolve(), args.output.resolve())
