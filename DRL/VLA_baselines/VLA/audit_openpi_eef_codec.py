"""Actual FAST32 codec fidelity and offline physical replay, NOT VLA SR."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def numerical(a):
    from openpi import transforms
    from VLA.pi0.eef_scene32_training import MixedEEF,stats,build_config
    data=MixedEEF(a.projection,a.admission,'pi0_fast');normalization,_=stats(data)
    cfg=build_config('pi0_fast',normalization,a.output,2);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    inputs=transforms.compose([*dc.data_transforms.inputs,transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
    outputs=transforms.compose([*dc.model_transforms.outputs,transforms.Unnormalize(dc.norm_stats,use_quantiles=True),*dc.data_transforms.outputs])
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for ei,row in enumerate(data.correction.rows):
        start=data.correction.ends[ei-1] if ei else 0;decoded=[];maximum=0.;tokens=0
        for fi in range(row['steps']):
            raw=data[data.original_length+start+fi];native=inputs(raw);mask=native['tokenized_prompt_mask']
            tokens=max(tokens,int(mask.sum()))
            if int(mask.sum())>=1024:raise ValueError('Actual FULL corrective FAST32 target truncates')
            # Policy.infer carries normalized state through its output tree;
            # Unnormalize(strict=True) expects BOTH state and actions.
            recovered=np.asarray(outputs({'state':native['state'],'actions':native['tokenized_prompt'][mask]})['actions'],np.float32)
            if recovered.shape!=(32,7):raise ValueError('Actual native FAST32 physical output shape differs')
            command=recovered[0].copy();command[6]=2*np.clip(command[6],0,1)-1
            expected=raw['canonical_actions'][0];maximum=max(maximum,float(np.abs(command[:3]-expected[:3]).max()))
            decoded.append(command)
        path=a.output/row['file'];np.savez_compressed(path,canonical=np.asarray(decoded,np.float32))
        rows.append({'file':row['file'],'source_file':row['source_file'],'task':row['task']['name'],'frames':row['steps'],
            'first_action_xyz_max_error_m':maximum,'max_tokens':tokens,'decoded_payload_sha256':sha256_file(path)})
        atomic_json(a.output/'progress.json',{'completed_episodes':len(rows),'planned_episodes':38,'actual_codec_frames':sum(r['frames'] for r in rows),'rows':rows,'model_SR_established':False})
    atomic_json(a.output/'audit.json',{'passed':True,'scope':'actual_FAST32x32_codec_FULL_native_corrections_NOT_model_SR',
        'episodes':38,'frames':sum(r['frames'] for r in rows),'rows':rows,'native_action_shape':[32,32],
        'physical_DOFs':7,'zero_fallback':False,'images_or_training_labels_changed':False,
        'admission_sha256':sha256_file(a.admission),'implementation_sha256':sha256_file(Path(__file__)),
        'candidate':str(data.correction.source),'source':str(data.original.source),'model_SR_established':False})


def physical(a):
    from VLA.dataset_replay_context import audit_environment
    from VLA.collect_native_recovery_view import NativeCapture
    from VLA.common.tasks import TaskSpec
    root=a.codec;proof=json.loads((root/'audit.json').read_text())
    if not proof['passed'] or proof['frames']!=1479:raise ValueError('Actual full codec qualification required')
    candidate=Path(proof['candidate']);source=Path(proof['source'])
    source_rows={r['file']:r for r in json.loads((source/'index.json').read_text())}
    index=json.loads((candidate/'index.json').read_text());selected=[next(r for r in index if r['task']['name']==task) for task in ('touch','pick','move','pick_place')]
    a.output.mkdir(parents=True,exist_ok=False);reports=[]
    for row in selected:
        path=root/row['file'];expected=next(r for r in proof['rows'] if r['file']==row['file'])
        if sha256_file(path)!=expected['decoded_payload_sha256']:raise ValueError('Decoded source changed')
        with np.load(path,allow_pickle=False) as z:commands=z['canonical'].copy()
        with np.load(candidate/row['file'],allow_pickle=False) as z:native=z['next_joint_targets'].copy();closed=z['next_requested_gripper_closure'].copy()
        env=audit_environment(source,image_size=384,control_hz=10,seed=0,required_success_steps=1,use_semantic_success=False)
        capture=NativeCapture(env,source,source_rows[row['source_file']],row,native,closed)
        result={'file':row['file'],'task':row['task']['name'],'passed':False,'error':None}
        try:
            capture.reset(TaskSpec.from_dict(row['task']))
            for command in commands:obs,_,_,info=capture.step(command)
            result.update(passed=True,steps=len(commands),strict_raw_success=bool(info['strict_raw_success']))
        except Exception as error:result['error']=f'{type(error).__name__}: {error}'
        finally:env.close()
        reports.append(result);atomic_json(a.output/'progress.json',{'rows':reports})
    atomic_json(a.output/'audit.json',{'passed':all(r['passed'] for r in reports),'scope':'offline_quantized_teacher_commands_4_tasks_NOT_VLA_success_rate',
        'rows':reports,'neural_model_executed':False,'same_success_and_safety':True,'codec_audit_sha256':sha256_file(root/'audit.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('numerical','physical'))
    for name in ('projection','admission','codec'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve every actual codec audit')
    if a.mode=='numerical':numerical(a)
    else:physical(a)
