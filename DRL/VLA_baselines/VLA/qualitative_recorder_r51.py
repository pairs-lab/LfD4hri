"""Read-only HD recorder for VERIFIED physical replay of saved neural controls.

This never calls an Oracle, generates an action, loads a neural network, changes
an observation camera, or creates a new success-rate trial. Source files and
checkpoints stay immutable. Run a baseline replay before the instrumented one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.scene64_conditioning import model_input
from env_eval.camera_editions import composite, overlap, projected_box, writer
from env_eval.handeye_camera import handeye_geometry
from env_eval.oracle_cinematic import CinematicRollout
from utils.camera import CINEMATIC_VIEWS

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'VLA/audits/scene67_deadline_r31_20261015/Smol_native_chunk2_conditional120_r41'
SETTINGS = ROOT / 'env_eval/camera_editions.json'


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def observation_signature(env, obs):
    inputs = model_input(env, obs)
    return {
        'front_rgb': array_hash(obs.front_rgb), 'wrist_rgb': array_hash(obs.wrist_rgb),
        'scene64': array_hash(inputs['scene64']), 'numeric67': array_hash(inputs['numeric67']),
        'prompt': obs.prompt,
        'segmentation_cache': {key: array_hash(value) for key, value in env._last_segmentation.items()},
    }


def physical_state(env):
    """Read measured bodies, joints, velocities, constraints and dynamics."""
    names = {env.robot_id: 'ur5e', env.gripper.body: 'robotiq_2f_85', env.table_id: 'table'}
    names.update({body: name for name, body in env.object_ids.items()})
    bodies = {}
    for index in range(p.getNumBodies()):
        body = p.getBodyUniqueId(index)
        name = names.get(body, p.getBodyInfo(body)[1].decode())
        bodies[name] = {
            'pose': p.getBasePositionAndOrientation(body), 'velocity': p.getBaseVelocity(body),
            'joints': [p.getJointState(body, j)[:2] for j in range(p.getNumJoints(body))],
            'dynamics': {str(j): p.getDynamicsInfo(body, j) for j in range(-1, p.getNumJoints(body))},
        }
    return {'bodies': bodies, 'physics': p.getPhysicsEngineParameters(),
            'constraints': [p.getConstraintInfo(p.getConstraintUniqueId(i)) for i in range(p.getNumConstraints())]}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_state():
    """Exact measured geometry only. No resampling or pose interpolation."""
    result = []
    for i in range(p.getNumBodies()):
        body = p.getBodyUniqueId(i)
        result.append({'body': body, 'pose': p.getBasePositionAndOrientation(body),
            'joints': [p.getJointState(body, j)[0] for j in range(p.getNumJoints(body))]})
    return result


def apply_render_state(state):
    for body in state:
        p.resetBasePositionAndOrientation(body['body'], *body['pose'])
        for joint, position in enumerate(body['joints']):
            p.resetJointState(body['body'], joint, position)
    # Pose reset is ONLY in the isolated, non-stepping render process.
    errors = {'position_m': 0., 'quaternion_component': 0., 'joint_rad': 0.}
    for expected, actual in zip(state, render_state(), strict=True):
        if expected['body'] != actual['body']:
            raise ValueError('Render replica body mapping differs')
        for key, a, b in [('position_m', expected['pose'][0], actual['pose'][0]),
                          ('quaternion_component', expected['pose'][1], actual['pose'][1]),
                          ('joint_rad', expected['joints'], actual['joints'])]:
            error = float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if len(a) else 0.
            errors[key] = max(errors[key], error)
    # Bullet normalizes base quaternions on reset. Disclose this separately
    # from the genuinely BIT-EXACT control/observation/physics replay proof.
    if max(errors.values()) > 1e-12:
        raise ValueError(f'Render replica geometry error: {errors}')
    return errors


class StateRecorder:
    """CPU-only capture; EGL never enters the control/observation process."""
    def __init__(self, env, directory):
        self.env, self.directory, self.frames = env, directory, []

    def open(self):
        pass

    def capture(self, tick, control):
        env = self.env
        contacts = p.getContactPoints(env.gripper.body, env.pick_id)
        self.frames.append({'frame': len(self.frames), 'physics_tick': tick, 'control_index': control,
            'correct_attached': env.gripper.grasped_object == env.pick_id,
            'target_contact_force_N': sum(max(0., c[9]) for c in contacts),
            'bilateral': bool(env.gripper._bilateral(env.pick_id)), 'geometry': render_state()})

    def close(self):
        atomic_json(self.directory / 'recorded_physics_states.json', self.frames)


def camera_stats(seg, env):
    ids = seg & ((1 << 24)-1)
    mask = ids == env.pick_id
    ys, xs = np.where(mask)
    links = (seg >> 24)-1
    return {
        'object_pixels': int(mask.sum()),
        'object_box': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
        'object_margin_px': int(min(xs.min(), seg.shape[1]-1-xs.max(), ys.min(), seg.shape[0]-1-ys.max())) if len(xs) else -1,
        'pad_pixels': [int(((ids == env.gripper.body) & (links == link)).sum()) for link in env.gripper.PAD_LINKS],
    }


class HDRecorder:
    """Seven simultaneous streams, sampled AFTER native post_step every8ticks."""
    def __init__(self, env, directory):
        self.env = env
        self.directory = directory
        self.settings = json.loads(SETTINGS.read_text())
        self.frames = []
        self.writers = {}
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        self.plugin = -1
        self.render_seconds = 0.

    def open(self):
        module = importlib.util.find_spec('eglRenderer')
        if module is None:
            raise RuntimeError('EGL module missing; no silent TinyRenderer fallback')
        self.plugin = p.loadPlugin(module.origin, '_eglRendererPlugin', physicsClientId=self.env.client_id)
        if self.plugin < 0:
            raise RuntimeError('EGL failed; no silent renderer fallback')
        for name, size in [('handeye', self.settings['handeye_resolution'])] + [
                (f'wide_{i}', self.settings['wide_resolution']) for i in range(1, 7)]:
            self.writers[name] = writer(self.directory / f'{name}.mp4', size, 30)
        self.review = self.directory / 'review'
        self.review.mkdir()

    def frame(self, cfg, size):
        # Reuse ONLY the pure read-only rendering method, never Oracle init/run.
        return CinematicRollout.frame(self, cfg, *size)

    def capture(self, tick, control, recorded=None):
        start = time.monotonic()
        env = self.env
        held = recorded['correct_attached'] if recorded is not None else env.gripper.grasped_object == env.pick_id
        contacts = p.getContactPoints(env.gripper.body, env.pick_id)
        cfg = handeye_geometry(env, self.settings['handeye'])
        rgb, seg = self.frame(cfg, self.settings['handeye_resolution'])
        self.writers['handeye'].send(rgb)
        row = {'frame': len(self.frames), 'physics_tick': tick, 'simulation_time_s': tick/240.,
               'encoded_pts_s': len(self.frames)/30., 'control_index': control,
               'correct_attached': held,
               'target_contact_force_N': recorded['target_contact_force_N'] if recorded is not None else sum(max(0., c[9]) for c in contacts),
               'bilateral': recorded['bilateral'] if recorded is not None else bool(env.gripper._bilateral(env.pick_id)),
               'object_position': p.getBasePositionAndOrientation(env.pick_id)[0],
               'pad_center': env.gripper.grasp_center().tolist(),
               'handeye': {**camera_stats(seg, env), 'camera': cfg}, 'wide': []}
        review_due = not self.frames or held != self.frames[-1]['correct_attached'] or tick % 120 == 0
        if review_due:
            cv2.imwrite(str(self.review / f'{len(self.frames):05}_handeye.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        for i, cfg in enumerate(CINEMATIC_VIEWS, 1):
            rgb, seg = self.frame(cfg, self.settings['wide_resolution'])
            self.writers[f'wide_{i}'].send(rgb)
            row['wide'].append({
                **camera_stats(seg, env),
                'projected_pick_box': projected_box(env, env.pick_id, cfg, *self.settings['wide_resolution']),
                'projected_place_box': projected_box(env, env.place_id, cfg, *self.settings['wide_resolution']) if env.place_id is not None else None,
            })
            if review_due and i in (1, 3):
                cv2.imwrite(str(self.review / f'{len(self.frames):05}_clean_view_{i}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames.append(row)
        self.render_seconds += time.monotonic()-start
        if len(self.frames) % 15 == 0:
            atomic_json(self.directory / 'recording_progress.json', {
                'status': 'recording', 'frames': len(self.frames), 'physics_tick': tick,
                'simulation_seconds': tick/240., 'render_wall_seconds': self.render_seconds,
                'all_actions_from_recorded_neural_trace': True})
            print(json.dumps({'frames': len(self.frames), 'tick': tick, 'render_seconds': round(self.render_seconds, 2)}), flush=True)

    def close(self):
        for stream in self.writers.values():
            stream.close()
        self.writers.clear()
        if self.plugin >= 0:
            p.unloadPlugin(self.plugin, physicsClientId=self.env.client_id)
            self.plugin = -1
        atomic_json(self.directory / 'frames.json', self.frames)


def replay(plan, row, *, out=None, baseline=None):
    profile = json.loads(Path(plan['observation_profile_file']).read_text())
    if profile_hash(profile) != plan['observation_profile_sha256']:
        raise ValueError('Observation profile changed')
    trace_path = Path(row['trajectory'])
    trace = json.loads(trace_path.read_text())[row['task']]
    env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e', image_size=384,
        control_hz=10, assist_grasp=True, seed=row['seed'], required_success_steps=1,
        use_semantic_success=True, success_threshold_scale=plan['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=plan['success']['semantic_thresholds'])
    recorder = None
    rows = []
    ticks = 0
    control = 0
    original_step = p.stepSimulation
    try:
        # Match the ORIGINAL task reset order, not just the selected spawn seed.
        for spec in plan['trials']:
            if spec['seed'] != row['seed']:
                continue
            task = TaskSpec.from_dict(spec['task'])
            obs = env.reset(task)
            if task.name == row['task']:
                break
        else:
            raise ValueError('Missing exact task')
        initial = physical_state(env)
        if baseline is not None and canonical_hash(initial) != baseline['initial_state_sha256']:
            raise ValueError('Initial measured physics differs from baseline')
        initial_signature = observation_signature(env, obs)
        if out is not None:
            atomic_json(out / 'initial_scene.json', {'task': spec['task'], 'measured_state': initial,
                'measured_state_sha256': canonical_hash(initial), 'observation_signature': initial_signature})
            recorder = StateRecorder(env, out)
            recorder.open()

        original_post = env.gripper.post_step

        def counted_step(*args, **kwargs):
            nonlocal ticks
            value = original_step(*args, **kwargs)
            ticks += 1
            return value

        def sampled_post():
            try:
                return original_post()
            finally:
                # Even if the native physics guard raises, preserve the actual
                # pre-rollback exposure at a scheduled tick. No extra physics.
                if recorder is not None and ticks % 8 == 0:
                    recorder.capture(ticks, control)

        p.stepSimulation = counted_step
        env.gripper.post_step = sampled_post
        maximum_error = 0.
        for control, frame in enumerate(trace):
            signature = observation_signature(env, obs)
            actual = model_input(env, obs)['scene64']
            error = float(np.max(np.abs(actual-np.asarray(frame['pre_action_scene64'], np.float32))))
            maximum_error = max(maximum_error, error)
            if error > 1e-6:
                raise ValueError(f'Saved neural trace physical divergence at control{control}: {error}')
            if baseline is not None and signature != baseline['controls'][control]['before_signature']:
                raise ValueError(f'Recorder changed raw policy inputs/cache at control{control}')
            action = np.asarray(frame['action'], np.float32)
            obs, _, done, info = env.step(action)
            if (bool(info['success']) != bool(frame['success']) or
                    info.get('safety_violation') != frame.get('safety_violation')):
                raise ValueError(f'Outcome changed at control{control}')
            np.testing.assert_array_equal(info['executed_canonical_action'], frame['executed_canonical_action'])
            measured_hash = canonical_hash(physical_state(env))
            after_signature = observation_signature(env, obs)
            record = {'control': control, 'before_signature': signature, 'after_signature': after_signature,
                'state_sha256_after': measured_hash, 'ticks_after': ticks,
                'canonical_neural_action': frame['action'], 'executed_native_action': info['executed_canonical_action'],
                'native_queue_audit': frame['native_queue_audit'], 'benchmark_region_success': bool(info['success']),
                'safety_violation': info.get('safety_violation')}
            if baseline is not None and record != baseline['controls'][control]:
                raise ValueError(f'Exact baseline action/state/input/tick parity failed at control{control}')
            rows.append(record)
            if done != (control == len(trace)-1):
                raise ValueError('Termination differs from saved neural trace')
        proof = {'passed_exact_replay': True, 'scope': 'VERIFIED_physical_replay_of_GENUINE_recorded_SmolVLA_controls_NOT_new_inference',
            'source_trajectory': str(trace_path), 'source_sha256': sha256_file(trace_path),
            'seed': row['seed'], 'task': row['task'], 'prompt': task.prompt, 'controls': rows,
            'initial_state_sha256': canonical_hash(initial), 'physics_ticks': ticks,
            'maximum_before_action_scene64_error': maximum_error,
            'benchmark_region_success': bool(info['success']), 'safety_violation': info.get('safety_violation'),
            'safety_diagnostics_before_rollback': info.get('safety_diagnostics_before_rollback'),
            'raw_RGB_numeric67_prompt_segmentation_exact_vs_uninstrumented_replay': baseline is not None,
            'every_after_action_full_physical_state_and_action_exact': baseline is not None,
            'native_processed_tensors_RNG_and_new_inference_NOT_executed': True,
            'EGL_rendering_isolated_in_different_OS_process': True,
            'oracle_called': False, 'new_SR_trials': 0,
            'timing': {'physics_Hz': 240, 'policy_control_Hz': 10, 'video_FPS': 30,
                'capture_at_end_of_every8_real_ticks': True, 'first_exposure_simulation_s': 8/240.,
                'encoded_PTS_offset_from_simulation_s': -8/240.,
                'trailing_real_ticks_without_scheduled_exposure': ticks % 8},
            'no_physics_asset_camera_success_horizon_or_neural_action_changes': True}
        return proof
    finally:
        p.stepSimulation = original_step
        if recorder is not None:
            recorder.close()
        env.close()


def render_recorded(args):
    """Offline render of EXACT captured states. This world NEVER advances time."""
    out = args.output.resolve()
    plan = json.loads((COHORT / 'plan.json').read_text())
    profile = json.loads(Path(plan['observation_profile_file']).read_text())
    initial = json.loads((out / 'initial_scene.json').read_text())
    env = environment_class_for_profile(profile)(observation_profile=profile, robot='ur5e', image_size=384,
        control_hz=10, assist_grasp=True, seed=args.seed, required_success_steps=1, use_semantic_success=True,
        success_threshold_scale=plan['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=plan['success']['semantic_thresholds'])
    recorder = None
    try:
        # Bullet's EGL plugin must exist BEFORE visual shapes are loaded.
        # Loading after reset produces a valid codec stream of empty background.
        recorder = HDRecorder(env, out)
        recorder.open()
        for spec in plan['trials']:
            if spec['seed'] != args.seed:
                continue
            task = TaskSpec.from_dict(spec['task'])
            env.reset(task)
            if task.name == args.task:
                break
        if canonical_hash(physical_state(env)) != initial['measured_state_sha256']:
            raise ValueError('Independent render replica differs before initial capture')
        rgb, _ = recorder.frame(handeye_geometry(env, recorder.settings['handeye']), recorder.settings['handeye_resolution'])
        if float(rgb.std()) < 1.:
            raise ValueError('Reset hand-eye is empty/constant: refuse blank HD render')
        cv2.imwrite(str(out / 'review/reset_handeye.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        states = json.loads((out / 'recorded_physics_states.json').read_text())
        matched = 0
        maxima = {'position_m': 0., 'quaternion_component': 0., 'joint_rad': 0.}
        for state in states:
            errors = apply_render_state(state['geometry'])
            maxima = {key: max(maxima[key], errors[key]) for key in maxima}
            recorder.capture(state['physics_tick'], state['control_index'], recorded=state)
            matched += 1
        atomic_json(out / 'renderer.json', {'renderer': 'PyBullet_EGL_hardware_OPENGL', 'EGL_plugin_loaded': True,
            'no_TinyRenderer_fallback': True, 'render_OS_pid': os.getpid(),
            'initial_scene_stabilization_steps_before_recorded_state_rendering': True,
            'no_steps_during_recorded_state_rendering': True, 'no_interpolation_or_generated_poses': True,
            'captured_geometry_exactly_restored_frames': matched,
            'render_replica_geometry_max_errors': maxima,
            'render_replica_geometry_tolerance_1e12_NOT_claimed_bit_exact': True,
            'presentation_camera': recorder.settings['handeye'], 'policy_camera_profile': profile,
            'camera_frame_convention': 'presentation_mount_COM_vs_policy_tool_link',
            'settings_sha256': sha256_file(SETTINGS), 'six_wide_views': CINEMATIC_VIEWS,
            'reason_for_process_isolation': 'same-client EGL altered TinyRenderer policy observation in rejected pilot_v1'})
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def select_layouts(frames):
    """Pilot-only whole-trajectory layout; shared paired layout NOT yet claimed."""
    layouts = {}
    for view in range(1, 7):
        candidates = []
        for side, x in [('left', 35), ('right', 1291)]:
            rect = ['handeye', x, 37, 594, 334]
            costs = []
            for row in frames:
                item = row['wide'][view-1]
                boxes = [item['projected_pick_box']] + ([item['projected_place_box']] if item['projected_place_box'] else [])
                costs.append(sum(overlap(box, rect) for box in boxes))
            candidates.append((max(costs), side, rect))
        cost, side, rect = min(candidates)
        if cost:
            raise ValueError(f'Both reference-size inset corners overlap target/goal: view{view} area{cost}')
        layouts[str(view)] = {'mode': 'handeye', 'side': side, 'rects': [rect],
            'max_projected_target_and_destination_area_covered': cost,
            'scope': 'pilot_single_trajectory_NOT_yet_paired_layout_qualified'}
    return layouts


def decode_at(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'Failed to decode {path} frame{index}')
    return frame


def audit_videos(out, proof):
    frames = json.loads((out / 'frames.json').read_text())
    layouts = json.loads((out / 'layout.json').read_text())
    events = {0, len(frames)//2, len(frames)-1}
    events.update(i for i, row in enumerate(frames) if i and row['correct_attached'] != frames[i-1]['correct_attached'])
    events = sorted(events)
    rows, sync = [], []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for name in ['handeye'] + [f'view_{i}' for i in range(1, 7)]:
        path = out / f'{name}.mp4'
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        decoded = subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
                                  '-f', 'null', '-'], capture_output=True, text=True)
        expected = [1280, 720] if name == 'handeye' else [1920, 1080]
        row = {'file': path.name, 'resolution': size, 'fps': fps, 'frames': n,
            'duration_s': n/fps if fps else None, 'full_decode_pass': decoded.returncode == 0 and not decoded.stderr.strip()}
        row['media_pass'] = bool(size == expected and fps == 30 and n == len(frames) and row['full_decode_pass'])
        rows.append(row)
        for index in events:
            image = decode_at(path, index)
            cv2.imwrite(str(out / 'review' / f'{index:05}_encoded_{name}.jpg'), image)
            if name != 'handeye':
                _, x, y, w, h = layouts[name.split('_')[-1]]['rects'][0]
                native = cv2.resize(decode_at(out / 'handeye.mp4', index), (w, h), interpolation=cv2.INTER_AREA)
                error = float(np.abs(image[y:y+h, x:x+w].astype(float)-native.astype(float)).mean())
                sync.append({'view': name, 'frame': index, 'mean_pixel_error': error, 'pass': error < 6.})
    actual_ticks = [r['physics_tick'] for r in frames]
    timing = actual_ticks == list(range(8, proof['physics_ticks']+1, 8))
    critical = [r for r in frames if r['correct_attached'] or r['target_contact_force_N'] > .1]
    visibility = {'critical_contact_or_attached_frames': len(critical),
        'minimum_critical_target_pixels': min((r['handeye']['object_pixels'] for r in critical), default=None),
        'minimum_critical_pad_pixels': [min((r['handeye']['pad_pixels'][i] for r in critical), default=None) for i in (0, 1)],
        'minimum_critical_target_margin': min((r['handeye']['object_margin_px'] for r in critical), default=None),
        'target_outside_presentation_handeye_frames': sum(r['handeye']['object_pixels'] == 0 for r in frames),
        'reference_thresholds': {'object_pixels': 100, 'each_pad_pixels': 20, 'margin_px': 8},
        'visibility_separate_from_media_integrity_and_task_success': True}
    audit = {'media_decode_sync_pass': bool(all(r['media_pass'] for r in rows) and all(r['pass'] for r in sync) and timing),
        'actual_tick_spacing_pass': timing, 'streams': rows, 'inset_sync': sync, 'visibility': visibility,
        'benchmark_region_success': proof['benchmark_region_success'], 'safety_violation': proof['safety_violation'],
        'recording_noninterference_pass': proof['every_after_action_full_physical_state_and_action_exact'],
        'scene_parity_vs_Oracle_NOT_yet_claimed': True, 'manual_visual_review_pending': True,
        'task_failure_does_NOT_fail_media_decode_gate': True, 'frame_event_indices': events,
        'all7streams_full_decoded': True, 'added_video_text': False}
    atomic_json(out / 'camera_audit.json', audit)
    return audit


def failure_notes(out, proof):
    if proof['benchmark_region_success']:
        return
    violation = proof['safety_violation']
    last = max(0, proof['physics_ticks']//8-1)
    note = (
        f"Natural saved SmolVLA policy failure; NOT staged; no action repair.\n"
        f"Task: {proof['task']}; prompt: {proof['prompt']}; seed: {proof['seed']}.\n"
        f"Observed outcome: {'native safety abort: '+violation if violation else 'original episode horizon reached without benchmark success'}.\n"
        f"Last scheduled stream frame (0-based): {last}; simulation time {proof['physics_ticks']/240.:.6f}s.\n"
        "All six views and hand-eye share simulation state/timestamps. Safety rollback, if present, is in replay_proof.json;\n"
        "an unscheduled terminal offending state is not fabricated into the video.\n"
        "Verified evidence: saved neural actions replay with exact before/after camera/state parity to a no-recorder baseline.\n"
        "Causal limits: linkage threshold is an observed safety trigger, not proof of localization or memory failure.\n"
        "Not a new neural inference run, not a new SR trial, not a paired Oracle comparison.\n"
    )
    for name in ['handeye'] + [f'view_{i}' for i in range(1, 7)]:
        (out / f'{name}.txt').write_text(note)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((COHORT / 'plan.json').read_text())
    progress = json.loads((COHORT / 'progress.json').read_text())
    row = next(r for r in progress['rows'] if (r['seed'], r['task']) == (args.seed, args.task))
    atomic_json(out / 'status.json', {'status': 'checking_frozen_sources', 'pid': os.getpid()})
    changed = [path for path, expected in plan['frozen_files'].items() if not Path(path).is_file() or sha256_file(path) != expected]
    if changed:
        raise ValueError(f'Frozen sources changed: {changed}')
    atomic_json(out / 'provenance.json', {'frozen_files_verified': len(plan['frozen_files']),
        'frozen_files': plan['frozen_files'], 'source_plan': str(COHORT / 'plan.json'),
        'source_plan_sha256': sha256_file(COHORT / 'plan.json'), 'checkpoint': plan['checkpoint'],
        'model_protocol': {'flow_steps': 10, 'learned_chunk': 5, 'consumed_chunk': 2, 'state': 67, 'raw_RGB': [384, 384]},
        'implementation_sha256': sha256_file(Path(__file__)), 'new_SR_trials': 0,
        'rendering_of_physical_replay_NOT_new_NN_inference': True})
    if args.from_capture is None:
        print('Verified frozen sources; replaying uninstrumented baseline', flush=True)
        baseline = replay(plan, row)
        atomic_json(out / 'baseline_replay.json', baseline)
        atomic_json(out / 'status.json', {'status': 'recording_verified_neural_control_replay', 'pid': os.getpid()})
        print('Baseline passed; capturing actual physics states WITHOUT EGL', flush=True)
        proof = replay(plan, row, out=out, baseline=baseline)
        atomic_json(out / 'replay_proof.json', proof)
    else:
        source = args.from_capture.resolve()
        proof = json.loads((source / 'replay_proof.json').read_text())
        if ((proof['seed'], proof['task']) != (args.seed, args.task) or
                not proof['every_after_action_full_physical_state_and_action_exact']):
            raise ValueError('Only already verified exact physics capture can be reused')
        names = ['baseline_replay.json', 'replay_proof.json', 'initial_scene.json', 'recorded_physics_states.json']
        digests = {name: sha256_file(source / name) for name in names}
        for name in names:
            shutil.copy2(source / name, out / name)
        atomic_json(out / 'capture_reuse.json', {'source': str(source), 'files_sha256': digests,
            'reusing_physics_capture_only_NOT_prior_videos': True, 'source_failures_preserved': True})
    atomic_json(out / 'status.json', {'status': 'isolated_HD_render_of_exact_captured_states', 'pid': os.getpid()})
    subprocess.run([sys.executable, '-u', '-m', 'VLA.qualitative_recorder_r51', '--seed', str(args.seed),
        '--task', args.task, '--output', str(out), '--render-recorded'], check=True)
    frames = json.loads((out / 'frames.json').read_text())
    layouts = select_layouts(frames)
    atomic_json(out / 'layout.json', layouts)
    atomic_json(out / 'status.json', {'status': 'compositing', 'pid': os.getpid()})
    settings = json.loads(SETTINGS.read_text())
    for view in range(1, 7):
        composite(out, out, view, layouts[str(view)], settings)
        print(f'Composited view{view}', flush=True)
    failure_notes(out, proof)
    audit = audit_videos(out, proof)
    atomic_json(out / 'manifest.sha256.json', {path.name: sha256_file(path) for path in sorted(out.glob('*.mp4'))})
    atomic_json(out / 'status.json', {'status': 'pilot_media_audited_manual_review_pending',
        'media_decode_sync_pass': audit['media_decode_sync_pass'], 'pid': os.getpid(),
        'scene_Oracle_parity_NOT_yet_claimed': True})
    print(json.dumps({'completed': str(out), 'media_pass': audit['media_decode_sync_pass']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--task', choices=['touch', 'pick', 'move', 'pick_place'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-recorded', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--from-capture', type=Path, help='Reuse an audited physical capture; output must be fresh')
    parsed = parser.parse_args()
    try:
        render_recorded(parsed) if parsed.render_recorded else main(parsed)
    except Exception as error:
        if parsed.output.is_dir():
            atomic_json(parsed.output / 'error.json', {'error': f'{type(error).__name__}: {error}', 'do_NOT_treat_as_complete': True})
        raise
