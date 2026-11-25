"""Package an actually repeat-verified Oracle/VLA pair with shared layout."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import SETTINGS, audit_videos, select_layouts
from env_eval.camera_editions import composite


def link(source, target):
    os.link(source, target)


def oracle_benchmark_evidence(outcome, task, frames):
    benchmark = outcome.get('benchmark_region_success_at_any_matched_10Hz_sample')
    if benchmark is not None:
        return benchmark, 'common_semantic_predicate_at24tick_boundaries'
    if task == 'touch':
        contact = any(r['physics_tick'] % 24 == 0 and r['target_contact_force_N'] > 0 for r in frames)
        return ((True, 'positive_contact_at_common10Hz_boundary_in_saved_physics_capture') if contact else
                (None, 'unknown_NOT_terminal_failure'))
    return (True if outcome['benchmark_region_success_at_terminal'] else None,
            'legacy_terminal_predicate_only_NOT_full_first_hit_log')


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    compared = ['initial_scene.json', 'trajectory.json', 'recorded_physics_states.json', 'outcome.json']
    if (args.oracle/'physics_commands.json').exists() and (args.oracle_repeat/'physics_commands.json').exists():
        compared.append('physics_commands.json')
    repeat = {name: sha256_file(args.oracle/name) == sha256_file(args.oracle_repeat/name) for name in compared}
    if not all(repeat.values()):
        raise ValueError('Oracle exact CPU repeat failed; do not render pair')
    oracle_outcome = json.loads((args.oracle/'outcome.json').read_text())
    oracle_initial = json.loads((args.oracle/'initial_scene.json').read_text())
    vla_initial = json.loads((args.vla/'initial_scene.json').read_text())
    if oracle_initial['measured_state_sha256'] != vla_initial['measured_state_sha256']:
        raise ValueError('Oracle/VLA initial physical state differs')
    seed = json.loads((args.vla/'replay_proof.json').read_text())['seed']
    task = oracle_initial['task']['name']
    source = root/'oracle_render_source'
    source.mkdir()
    for name in ['initial_scene.json', 'recorded_physics_states.json', 'protocol.json', 'outcome.json']:
        shutil.copy2(args.oracle/name, source/name)
    atomic_json(root/'status.json', {'status': 'rendering_repeat_verified_shared_world_Oracle', 'pid': os.getpid()})
    subprocess.run([sys.executable, '-u', '-m', 'VLA.qualitative_recorder_r51', '--seed', str(seed),
        '--task', task, '--output', str(source), '--render-recorded'], check=True)
    vla_frames = json.loads((args.vla/'frames.json').read_text())
    oracle_frames = json.loads((source/'frames.json').read_text())
    layouts = select_layouts(vla_frames+oracle_frames)
    for layout in layouts.values():
        layout['scope'] = 'ONE_SHARED_layout_checked_over_BOTH_complete_trajectories'
    settings = json.loads(SETTINGS.read_text())
    outcomes = {}
    for method, recorded in [('Oracle', source), ('SmolVLA', args.vla)]:
        dest = root/method
        dest.mkdir()
        for name in ['handeye.mp4', 'frames.json', 'initial_scene.json', 'renderer.json']:
            link(recorded/name, dest/name)
        atomic_json(dest/'layout.json', layouts)
        for view in range(1, 7):
            composite(recorded, dest, view, layouts[str(view)], settings)
        (dest/'review').mkdir()
        if method == 'SmolVLA':
            proof = json.loads((args.vla/'replay_proof.json').read_text())
            for name in ['replay_proof.json', 'provenance.json']:
                shutil.copy2(args.vla/name, dest/name)
            for note in args.vla.glob('*.txt'):
                shutil.copy2(note, dest/note.name)
        else:
            # Score a first-hit predicate, not Oracle's pose AFTER retracting.
            # Older captures did not log the common 10Hz predicate. A measured
            # positive target force at a 24-tick boundary certifies touch; do
            # not turn a negative terminal predicate into a failure score.
            benchmark, evidence = oracle_benchmark_evidence(oracle_outcome, task, oracle_frames)
            proof = {'physics_ticks': oracle_outcome['recorded_physics_ticks'],
                'benchmark_region_success': benchmark,
                'safety_violation': oracle_outcome['result'].get('error'),
                # VLA-specific before/after-action signature test was NOT run for
                # Oracle. Its whole CPU capture and controls were repeated instead.
                'every_after_action_full_physical_state_and_action_exact': False}
            atomic_json(dest/'Oracle_repeat_proof.json', {'all_files_byte_exact': repeat,
                'source': str(args.oracle.resolve()), 'repeat': str(args.oracle_repeat.resolve()),
                'scope': 'all30FPS_physics_snapshots_and_controller_trajectory_and_outcome',
                'genuine_Oracle_NOT_VLA': True, 'NN_inference_run': False})
            shutil.copy2(args.oracle/'protocol.json', dest/'protocol.json')
        audit = audit_videos(dest, proof)
        if method == 'Oracle':
            audit['recording_noninterference_pass'] = all(repeat.values())
            audit['noninterference_evidence'] = 'whole CPU capture repeated byte-exact; not VLA tensor/RNG comparison'
            audit['benchmark_predicate_evidence'] = evidence
            audit['terminal_predicate_after_retraction_NOT_used_as_failure_score'] = True
        audit['scene_parity_vs_Oracle_NOT_yet_claimed'] = False
        audit['scene_initial_parity_pass'] = True
        audit['shared_layout_over_both_trajectories_pass'] = True
        observed = json.loads((dest/'frames.json').read_text())
        if method == 'Oracle':
            qualitative_complete = bool(oracle_outcome['result'].get('success', False))
        elif task == 'pick_place':
            # Legacy region success while still holding is NOT a full put.
            # No unsupported claim of support/stability after release, either.
            qualitative_complete = False if observed[-1]['correct_attached'] or not proof['benchmark_region_success'] else None
        elif task == 'touch':
            qualitative_complete = bool(proof['benchmark_region_success'] and any(r['target_contact_force_N'] > .03 for r in observed))
        else:
            qualitative_complete = bool(proof['benchmark_region_success'] and any(r['correct_attached'] for r in observed))
        audit['qualitative_task_complete'] = qualitative_complete
        audit['qualitative_completion_NOT_same_as_legacy_region_success'] = True
        atomic_json(dest/'camera_audit.json', audit)
        if qualitative_complete is not True:
            additional_note = ('Selected qualitative recording; no staged fault.\n'
                f'Method: {method}; task: {task}; seed: {seed}.\n'
                f'Benchmark region success: {proof["benchmark_region_success"]}; full qualitative completion: {qualitative_complete}.\n'
                f'Actual final scheduled frame: {len(observed)-1}; final exposure time: {observed[-1]["simulation_time_s"]:.6f}s.\n'
                f'Safety/termination evidence: {proof.get("safety_violation")}.\n'
                f'Correct attachment observed: {any(r["correct_attached"] for r in observed)}; attached at final exposure: {observed[-1]["correct_attached"]}.\n'
                'For put, region success with an attached object is NOT release-and-stable-placement.\n'
                'See source replay/Oracle repeat proofs, frames.json and camera_audit.json.\n'
                'No claim of memory loss or model-intrinsic weakness follows from this footage.\n')
            for name in ['handeye'] + [f'view_{i}' for i in range(1, 7)]:
                note_path = dest/(name+'.txt')
                existing = note_path.read_text() + '\n' if note_path.exists() else ''
                note_path.write_text(existing + additional_note)
        outcomes[method] = {'media_pass': audit['media_decode_sync_pass'],
            'benchmark_region_success': proof['benchmark_region_success'],
            'qualitative_task_complete': qualitative_complete,
            'duration_s': audit['streams'][0]['duration_s']}
        atomic_json(dest/'manifest.sha256.json', {p.name: sha256_file(p) for p in sorted(dest.glob('*.mp4'))})
    atomic_json(root/'PAIR_AUDIT.json', {'initial_state_sha256': vla_initial['measured_state_sha256'],
        'scene_initial_parity_pass': True, 'Oracle_repeat_files_byte_exact': repeat,
        'shared_layout_pass': True, 'outcomes': outcomes, 'new_SR_trials': 0,
        'Oracle_is_adapted_to_frozen_VLA_assets_and_guard_NOT_original_reference_environment': True,
        'Smol_is_verified_saved_neural_control_replay_NOT_fresh_inference': True,
        'manual_paired_visual_review_pending': True, 'full11_scene_suite_complete': False})
    sections = []
    for method in ('Oracle', 'SmolVLA'):
        sections.append(f'<section><h2>{method}</h2><video controls preload="metadata" width="800" src="{method}/view_1.mp4"></video><p>' +
            ' · '.join(f'<a href="{method}/view_{i}.mp4">view{i}</a>' for i in range(1, 7)) +
            f' · <a href="{method}/handeye.mp4">native hand-eye</a></p></section>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Actual paired pilot</title>' +
        '<h1>UR5e + Robotiq — paired qualitative pilot</h1><p>' + html.escape(oracle_initial['task']['prompt']) +
        '</p><p>SmolVLA: verified replay of genuine saved neural controls. Oracle: existing feedback controller adapted to the SAME frozen physical world; not learned RL. No speed-up, no new SR estimate.</p>' + ''.join(sections))
    atomic_json(root/'status.json', {'status': 'paired_media_audited_manual_review_pending', 'pid': os.getpid(), 'outcomes': outcomes})
    print(json.dumps({'paired_pilot': str(root), 'outcomes': outcomes}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'oracle-repeat', 'vla', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
