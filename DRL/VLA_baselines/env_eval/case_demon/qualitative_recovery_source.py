"""Recover a natural drop/reacquisition from immutable actual X-VLA traces.

This is new qualitative capture of an EXISTING neural episode, not new neural
inference, a new SR trial or induced failure. The saved actual action sequence
is replayed twice through the original pinned physics/RGB binding. The Oracle
counterpart is explicitly a scripted demonstration of the exact same scene.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from .qualitative_figure import ROOT,OUTPUT,read,write,digest

SEED=20271063
SIM=ROOT/'VLA/.venvs/qualitative_r57/bin/python'
TRACE=ROOT/f'VLA/audits/xvla800_r59_20261017/episodes/test_seed{SEED}/trajectory.json'


def main(args):
    out=args.output.resolve();dest=out/'additional_recovery_source'
    dest.mkdir(parents=True,exist_ok=True)
    trace=read(TRACE)['pick_place']
    lost=[i for i in range(1,len(trace)) if trace[i-1]['after']['correct_object_held']
          and not trace[i]['after']['correct_object_held']]
    assert lost and trace[-1]['success'] and any(r['after']['correct_object_held'] for r in trace[lost[0]+1:])
    candidate={'seed':SEED,'trajectory':str(TRACE),'trajectory_sha256':digest(str(TRACE)),
               'cohort':str(ROOT/'VLA/audits/xvla800_r59_20261017'),
               'reason':'actual_saved_neural_drop_then_reacquisition; must pass physical replay and visual audit',
               'NOT_new_neural_trial':True,'NOT_induced_disturbance':True}
    write(dest/'candidate.json',[candidate])
    jobs=[('VLA.qualitative_r57.replay',['--seed',SEED,'--shortlist',dest/'candidate.json',
            '--output',dest/'XVLA_capture','--terminal-release'],'XVLA_capture/replay_proof.json'),
          ('VLA.qualitative_r57.oracle',['--seed',SEED,'--paired-capture',dest/'XVLA_capture',
            '--output',dest/'Oracle_capture','--fault','none'],'Oracle_capture/outcome.json'),
          ('VLA.qualitative_r57.events',['--capture',dest/'XVLA_capture'],'XVLA_capture/content_evidence.json'),
          ('VLA.qualitative_r57.events',['--capture',dest/'Oracle_capture'],'Oracle_capture/content_evidence.json')]
    for index,(module,options,marker) in enumerate(jobs):
        if (dest/marker).exists():
            continue
        write(dest/'status.json',{'stage':module,'running':True,'original_trace_sha256':candidate['trajectory_sha256']})
        log=dest/f'step{index}_{module.rsplit(".",1)[-1]}.log'
        with log.open('w') as stream:
            result=subprocess.run([str(SIM),'-u','-m',module,*map(str,options)],cwd=ROOT,
                                  stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(dest/'status.json',{'stage':module,'failed':True,'log':str(log),
                                      'NOT_a_new_model_failure_or_SR_trial':True})
            raise RuntimeError(f'{module} exited {result.returncode}; see {log}')
    proof=read(dest/'XVLA_capture/replay_proof.json');ev=read(dest/'XVLA_capture/content_evidence.json')
    oracle=read(dest/'Oracle_capture/content_evidence.json')
    valid=(proof['passed_exact_replay'] and proof['every_after_action_full_physical_state_and_action_exact']
           and proof['raw_observation_segmentation_and_prompt_bit_exact_vs_baseline']
           and ev['full_put_complete'] and oracle['full_put_complete']
           and any(e['mode']=='verified_in_transit_drop' for e in ev['events']))
    write(dest/'capture_qualification.json',{'passed':valid,'source':candidate,
            'VLA_full_put':ev['full_put_complete'],'Oracle_full_put':oracle['full_put_complete'],
            'VLA_events':ev['events'],'scene_id':14,'manual_visual_review_pending':True,
            'existing_quantitative_episode_not_new_trial':True})
    if not valid:raise ValueError('Natural recovery was not physically qualified')
    write(dest/'status.json',{'stage':'captured_exact_physics_and_RPC','passed':True,'needs_render':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    main(p.parse_args())
