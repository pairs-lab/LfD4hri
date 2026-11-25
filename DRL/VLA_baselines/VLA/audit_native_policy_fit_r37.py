"""Bounded TRAIN-action probes of the actual already-running native VLAs.

No new GPU model, simulator controls, training, action correction or task SR.
Each observation contains the real source images, measured proprio and all
causal64 fields. Gold labels are local only, NEVER sent to the model server.
Explicit image history and request-local noise/greedy decoding are required;
the source implementations are checked before any auxiliary request.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import augment_prompt, validate_scene, PROMPT_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'VLA/audits/scene67_deadline_r31_20261015'
PROJECTION = ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
TASKS = ('touch', 'pick', 'move', 'pick_place')
PORTS = {'rdt':5812, 'cogact':5813, 'pi0':5821, 'pi0_fast':5822, 'openvla':5831}


def select_frames(actions, states):
    """Predeclared motion/close/held strata; never inspect neural outcomes."""
    actions, states = np.asarray(actions), np.asarray(states)
    if actions.shape != (len(states),7) or states.shape[1:] != (64,):
        raise ValueError('Actual complete TRAIN observation/command timeline required')
    xy = np.linalg.norm(actions[:,:2], axis=1)
    stages = [('xy_alignment', np.flatnonzero((xy>.005)&(states[:,17]==0))),
        ('near_unheld_close', np.flatnonzero((actions[:,6]>.5)&(states[:,17]==0))),
        ('verified_hold_transport', np.flatnonzero((states[:,17]==1)&(np.linalg.norm(actions[:,:3],axis=1)>.005)))]
    result=[];seen=set()
    for stage,ids in stages:
        if len(ids):
            # The first substantial control, not a post-success observation.
            index=int(ids[0])
            if index not in seen:result.append((stage,index));seen.add(index)
    if not result:raise ValueError('No substantive labelled motion; do not substitute a stationary frame')
    return result


def to_canonical(values, kind):
    value=np.asarray(values,np.float32).copy()
    if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('Malformed genuine neural command')
    if kind=='native_delta_ee_and_open_gripper':value[6]=1-2*value[6]
    elif kind!='canonical_delta_ee_7d':raise ValueError('Not an EEF action representation')
    return value


def metrics(prediction, kind, gold, joints):
    value=np.asarray(prediction,np.float32)
    if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('Invalid actual native prediction')
    if kind=='absolute_joint_targets_and_closure':
        target=np.asarray(joints,np.float32)
        error=(value[:6]-target[:6]+np.pi)%(2*np.pi)-np.pi
        return {'native_joint_MAE_rad':float(np.abs(error).mean()),
            'native_joint_max_error_rad':float(np.abs(error).max()),
            'gripper_sign_matches':bool((value[6]>.5)==(gold[6]>.0)),
            'units_NOT_comparable_to_EEF_mm':True}
    value=to_canonical(value,kind);target=np.asarray(gold,np.float32)
    xyden=np.linalg.norm(value[:2])*np.linalg.norm(target[:2])
    return {'XYZ_error_mm':float(np.linalg.norm(value[:3]-target[:3])*1000),
        'XY_direction_cosine':float(np.dot(value[:2],target[:2])/xyden) if xyden>1e-9 else None,
        'yaw_error_rad':float(abs(value[5]-target[5])),
        'gripper_sign_matches':bool((value[6]>0)==(target[6]>0))}


def fixture(output):
    from VLA.privileged_native_trial import png
    from VLA.common.native_action_fields import native_gripper_closure
    output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((PROJECTION/'audit.json').read_text());source=Path(audit['source'])
    if not audit.get('passed') or audit.get('pilot_only') or audit.get('split')!='train':
        raise ValueError('Complete TRAIN projection only')
    rows=json.loads((source/'index.json').read_text());manifest=json.loads((source/'dataset_manifest.json').read_text())
    if manifest.get('split')!='train':raise ValueError('No validation/development/final observations in this probe')
    if sha256_file(source/'index.json')!=audit['source_index_sha256']:raise ValueError('TRAIN index changed')
    proofs={r['file']:r for r in audit['rows']};selected=[]
    for task in TASKS:
        names=set()
        for row in rows:
            if row['task']['name']!=task or row.get('paired_scene_source') or row.get('recovery_start_actions_not_training_labels'):
                continue
            if row['task']['pick_name'] in names:continue
            if not row.get('success'):raise ValueError('Unexpected failed original TRAIN demonstration')
            names.add(row['task']['pick_name']);selected.append(row)
            if len(names)==2:break
        if len(names)!=2:raise ValueError('Need two predeclared distinct TRAIN targets per task')
    items=[];bound={str(p):sha256_file(p) for p in (PROJECTION/'audit.json',source/'index.json',source/'dataset_manifest.json',Path(__file__))}
    for row in selected:
        path=source/row['file'];statepath=PROJECTION/'states'/row['file'];proof=proofs[row['file']]
        if sha256_file(path)!=proof['source_payload_sha256'] or sha256_file(statepath)!=proof['state_payload_sha256']:
            raise ValueError('Selected actual source bytes changed')
        bound.update({str(path):proof['source_payload_sha256'],str(statepath):proof['state_payload_sha256']})
        with np.load(statepath,allow_pickle=False) as z:states=z['state'].copy()
        with np.load(path,allow_pickle=False) as z:
            actions=z['canonical_actions'];choices=select_frames(actions,states)
            for stage,frame in choices:
                raw=validate_scene(states[frame]);instruction=str(z['prompt'].item());prompt=augment_prompt(instruction,raw)
                np.testing.assert_allclose(raw[:6],z['joint_positions'][frame],rtol=0,atol=1e-6)
                previous=max(frame-1,0)
                native=np.r_[z['joint_positions'][frame].astype(np.float32),np.float32(z['gripper_open'][frame]),np.zeros(9,np.float32)]
                if abs(float(native[6])-(1-float(raw[6])))>1e-6:raise ValueError('Measured gripper-open/closed sensor disagrees')
                payload={'schema':PROMPT_SCHEMA,'instruction':instruction,'prompt':prompt,'scene64':raw.tolist(),
                    'front_rgb_png':png(z['front_rgb'][frame]),'wrist_rgb_png':png(z['wrist_rgb'][frame]),
                    'previous_front_rgb_png':png(z['front_rgb'][previous]),'previous_wrist_rgb_png':png(z['wrist_rgb'][previous]),
                    'native_proprio':native.tolist(),'seed':20263701,'inference_index':frame}
                item={'number':len(items),'task':row['task']['name'],'target':row['task']['pick_name'],
                    'stage':stage,'source_file':str(path),'source_frame':frame,'TRAIN_only':True,
                    'canonical_gold_LOCAL_ONLY':actions[frame].astype(np.float32).tolist(),
                    'native_gold_LOCAL_ONLY':np.r_[z['next_joint_targets'][frame],native_gripper_closure(z)[frame]].astype(np.float32).tolist(),
                    'payload':payload}
                items.append(item)
    atomic_json(output/'requests_with_LOCAL_gold.json',items)
    atomic_json(output/'audit.json',{'passed':True,'scope':'predeclared_TRAIN_substantive_actions_NOT_model_SR',
        'observations':len(items),'episodes':len(selected),'strata':dict(Counter(i['stage'] for i in items)),
        'tasks':dict(Counter(i['task'] for i in items)),'source':str(source),'all_causal64_and_real_previous_RGB':True,
        'gold_sent_to_server':False,'simulator_controls_executed':0,'GPU_model_loaded':False,
        'selection_uses_neural_results':False,'validation_dev_final_used':False,
        'request_file_sha256':sha256_file(output/'requests_with_LOCAL_gold.json'),'bound_files':bound})


def find_server(model):
    module={'rdt':'VLA.rdt.scene67_server','cogact':'VLA.cogact.scene67_server',
        'pi0':'VLA.pi0.eef_scene32_server','pi0_fast':'VLA.pi0.eef_scene32_server',
        'openvla':'VLA.openvla.scene64_server'}[model]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            tokens=(proc/'cmdline').read_bytes().split(b'\0')
            if module.encode() not in tokens or os.readlink(proc/'cwd')!=str(ROOT):continue
            port=tokens[tokens.index(b'--port')+1].decode()
            if port!=str(PORTS[model]):continue
            cp=Path(tokens[tokens.index(b'--checkpoint')+1].decode())
            output=Path(tokens[tokens.index(b'--output')+1].decode())
            # Only long-fit generations, not the tiny numerical qualifications.
            if model in ('rdt','cogact') and '2000' not in str(output):continue
            if model in ('pi0','pi0_fast'):
                if tokens[tokens.index(b'--model')+1].decode()!=model or '2000' not in str(output):continue
            if model=='openvla' and ('OpenVLA_balanced_recovery_r34_' not in str(output) or output.name!='server1000_runtime'):continue
            return {'pid':int(proc.name),'checkpoint':str(cp),'output':str(output),'module':module}
        except (OSError,ValueError,IndexError):continue
    return None


def probe(model,server,fixtures,output):
    import requests
    if output.exists():raise FileExistsError('Never overwrite an actual probe attempt')
    files={'rdt':('VLA/rdt/scene67_server.py','VLA/rdt/server.py'),
        'cogact':('VLA/cogact/scene67_server.py',),
        'pi0':('VLA/pi0/eef_scene32_server.py',),
        'pi0_fast':('VLA/pi0/eef_scene32_server.py','VLA/third_party/openpi/src/openpi/models/pi0_fast.py'),
        'openvla':('VLA/openvla/scene64_server.py',)}[model]
    bound={str(ROOT/name):sha256_file(ROOT/name) for name in files}
    if model=='rdt':
        text=(ROOT/files[0]).read_text()
        if 'torch.random.fork_rng' not in text or 'previous_front_rgb_png' not in (ROOT/files[1]).read_text():
            raise ValueError('No safe request-local RNG/explicit history contract')
    elif model=='cogact' and 'torch.random.fork_rng' not in (ROOT/files[0]).read_text():raise ValueError('CogACT RNG is not isolated')
    elif model=='pi0' and "kwargs['noise']" not in (ROOT/files[0]).read_text():raise ValueError('Pi0 explicit noise required')
    elif model=='pi0_fast':
        native=(ROOT/files[1]).read_text()
        if 'temperature: float = 0.0' not in native or 'jnp.argmax' not in native:raise ValueError('Only deterministic native greedy FAST may be probed')
    url=f'http://127.0.0.1:{PORTS[model]}';session=requests.Session()
    metadata=session.get(url+'/metadata',timeout=3).json()
    if metadata.get('model')!=model or metadata.get('checkpoint')!=server['checkpoint'] or metadata.get('actual_feedback_fields')!=64:
        raise ValueError('Wrong running server/model/full-feedback generation')
    proof=json.loads((fixtures/'audit.json').read_text())
    requests_path=fixtures/'requests_with_LOCAL_gold.json'
    if not proof['passed'] or sha256_file(requests_path)!=proof['request_file_sha256']:raise ValueError('Fixture changed')
    for path,digest in proof['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Qualified selected TRAIN source or probe implementation changed')
    items=json.loads(requests_path.read_text());output.mkdir(parents=True);rows=[];first=None;error=None;repeated=None
    def save(status):
        ee=[r['metrics']['XYZ_error_mm'] for r in rows if 'XYZ_error_mm' in r['metrics']]
        joint=[r['metrics']['native_joint_MAE_rad'] for r in rows if 'native_joint_MAE_rad' in r['metrics']]
        atomic_json(output/'audit.json',{'status':status,'model':model,'server':server,'metadata':metadata,
            'completed_requests':len(rows),'planned_requests':len(items)+1,'rows':rows,'error':error,
            'repeat_input_max_abs_difference':repeated,'median_XYZ_error_mm':float(np.median(ee)) if ee else None,
            'median_native_joint_MAE_rad':float(np.median(joint)) if joint else None,
            'gripper_sign_correct':sum(r['metrics']['gripper_sign_matches'] for r in rows),
            'scope':'small_selected_TRAIN_fit_and_repeatability_probe_NOT_SR_or_generalization',
            'noise_realizations_between_prompts_NOT_equal_so_NOT_causal_language_ablation':True,
            'gold_labels_never_sent':True,'simulator_controls_executed':0,'new_GPU_model_loaded':False,
            'fixture_audit_sha256':sha256_file(fixtures/'audit.json'),'bound_files':bound})
    save('running_actual_native_requests')
    try:
        for item in items:
            response=session.post(url+'/act',json=item['payload'],timeout=900);response.raise_for_status();value=response.json()
            if value.get('action_source')!='native_neural_model' or value.get('prompt_sha256')!=hashlib.sha256(item['payload']['prompt'].encode()).hexdigest():
                raise ValueError('Wrong neural source/current prompt identity')
            actions=np.asarray(value['actions'],np.float32)
            if actions.ndim!=2 or actions.shape[1]!=7 or not len(actions) or not np.isfinite(actions).all():raise ValueError('Malformed native action chunk')
            if first is None:first=actions.copy()
            result={k:v for k,v in item.items() if k!='payload'}
            result.update(actual_first_neural_action=actions[0].tolist(),action_kind=value['action_kind'],
                metrics=metrics(actions[0],value['action_kind'],item['canonical_gold_LOCAL_ONLY'],item['native_gold_LOCAL_ONLY']))
            rows.append(result);save('running_actual_native_requests')
        response=session.post(url+'/act',json=items[0]['payload'],timeout=900);response.raise_for_status();value=response.json()
        if (value.get('action_source')!='native_neural_model'
                or value.get('prompt_sha256')!=hashlib.sha256(items[0]['payload']['prompt'].encode()).hexdigest()):
            raise ValueError('Repeat not the same native observation/prediction')
        again=np.asarray(value['actions'],np.float32)
        if again.shape!=first.shape or not np.isfinite(again).all():raise ValueError('Repeat has incompatible output')
        repeated=float(np.max(np.abs(first-again)))
        if repeated!=0:raise ValueError('Same native request is not exactly repeatable; investigate before stability claims')
        if any(sha256_file(p)!=digest for p,digest in bound.items()):raise ValueError('Source changed during probe')
        save('completed_actual_native_fit_probe_NOT_task_success')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_or_interrupted_probe_NOT_zero_SR');raise
    finally:session.close()


def watch(a):
    a.output.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+a.wait_hours*3600;done={}
    while time.monotonic()<deadline:
        for model in PORTS:
            if model in done:continue
            server=find_server(model)
            if server is None:continue
            import requests
            try:
                r=requests.get(f'http://127.0.0.1:{PORTS[model]}/metadata',timeout=2)
                if not r.ok or r.json().get('checkpoint')!=server['checkpoint']:continue
            except (requests.RequestException,ValueError):continue
            try:probe(model,server,a.fixture,a.output/model);done[model]='actual_probe_completed_NOT_SR'
            except Exception as exc:done[model]=f'probe_failed_preserved_NOT_zero_SR: {type(exc).__name__}: {exc}'
        atomic_json(a.output/'status.json',{'pid':os.getpid(),'status':'completed' if len(done)==len(PORTS) else 'waiting_actual_long_fit_servers',
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'models':done,'pending_not_failed':sorted(set(PORTS)-set(done)),
            'no_training_or_evaluator_modified':True,'no_simulator_commands':True,'no_additional_GPU_model':True,
            'deadline_remaining_seconds':max(0,deadline-time.monotonic())})
        if len(done)==len(PORTS):return
        time.sleep(30)
    atomic_json(a.output/'deadline.json',{'completed':done,'pending_NOT_zero_SR':sorted(set(PORTS)-set(done)),'finite_wait_exhausted':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('fixture','watch'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path)
    p.add_argument('--wait-hours',type=float,default=40);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=48:raise ValueError('Finite diagnostic only')
    if a.launch:
        if a.mode!='watch' or a.output.exists():raise ValueError('Only one fresh detached watcher')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.audit_native_policy_fit_r37','watch','--output',str(a.output.resolve()),
                '--fixture',str(a.fixture.resolve()),'--wait-hours',str(a.wait_hours)],cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(a.output.with_name(a.output.name+'_launch.json'),{'pid':child.pid,'no_GPU_model_or_robot_controller':True});print(child.pid)
    elif a.mode=='fixture':fixture(a.output.resolve())
    else:watch(a)
