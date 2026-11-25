"""Synthetic fixtures stay in TemporaryDirectory; never counted as robot trials."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import numpy as np

from .common import (CP,FINAL,SMOKE,TASKS,Heartbeat,retryable,approved,
    native_s_callbacks,durable_json,read)
from .evaluate import check_pending_journal
from VLA.autorefinement_r97.evaluate import ledger
from VLA.autonomy_r83.common import metrics


def fixture_progress(count=0):
    ordered=[(s,t) for s in FINAL for t in TASKS]
    rows=[{'seed':s,'task':t,'success':False,'steps':1,'trajectory':'synthetic_NOT_experiment',
        'safety_violation':None} for s,t in ordered[:count]]
    return {'rows':rows,'per_action':metrics(rows),'completed_trials':len(rows),'planned_trials':800,
        'status':'integration_error_NOT_model_SR','cohort_role':'fresh_secondary_final800',
        'motor_controls':len(rows),'synthetic_fixture_NOT_experiment':True}


class Contracts(unittest.TestCase):
    def test_four_hours_of_live_queries_never_stall(self):
        h=Heartbeat(0)
        for now in range(0,4*3600,30):
            self.assertLess(h.observe((now//30,),now),900)

    def test_unchanged_episode_count_with_live_query_is_healthy(self):
        h=Heartbeat(0)
        for now in range(0,1200,10):self.assertEqual(h.observe((0,now),now),0)

    def test_stall_detected_only_after_no_heartbeat(self):
        h=Heartbeat(0);h.observe(('unchanged',),0)
        self.assertLess(h.observe(('unchanged',),899),900)
        self.assertGreater(h.observe(('unchanged',),901),900)

    def test_real_clock_backward_not_used(self):
        h=Heartbeat(100);h.observe((1,),100)
        self.assertEqual(h.observe((1,),110),10)

    def test_integrity_error_never_retried_even_after_death(self):
        for text in ('ValueError: bad receipt','TypeError: invalid','Native map pixel receipt mismatch',
            'prefix differs','Bound file changed'):
            self.assertFalse(retryable(text,abrupt=True,stalled=True))

    def test_transient_errors_retry(self):
        for text in ('ConnectionError','ReadTimeout','CUDA out of memory','Resource interruption'):
            self.assertTrue(retryable(text))
        self.assertTrue(retryable('',abrupt=True));self.assertTrue(retryable('',stalled=True))
        self.assertFalse(retryable('task safety abort; normal task failure'))

    def test_fixed_checkpoint_and_full_seed_block_authorized(self):
        method={'family':'vlanext_s','checkpoint':str(CP),'R98_frozen_before_any_evaluation':True,
            'actual_checkpoint_optimizer_updates':9216,'frozen_for_user_requested_benchmark':True,
            'benchmark_authorization':{'user_requested_direct_postfit800':True}}
        cohort={'role':'fresh_secondary_final800','seeds':FINAL}
        self.assertTrue(approved(method,cohort))
        self.assertFalse(approved({**method,'family':'dita'},cohort))
        self.assertFalse(approved(method,{**cohort,'seeds':FINAL[:-1]}))
        self.assertFalse(approved({**method,'actual_checkpoint_optimizer_updates':8192},cohort))
        self.assertFalse(set(SMOKE)&set(FINAL))

    def test_partial_ledgers_preserve_failed_outcomes(self):
        for count in (0,1,3,4,799,800):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'progress.json';path.write_text(json.dumps(fixture_progress(count)))
                data,rows,keys=ledger(path,{'seeds':FINAL})
                self.assertEqual(len(rows),count)
                self.assertEqual(sum(r['success'] for r in rows),0)

    def test_duplicate_missing_or_reordered_rows_rejected(self):
        base=fixture_progress(4)
        for rows in ([base['rows'][0]]*4,list(reversed(base['rows'])),base['rows'][1:]):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'p.json';path.write_text(json.dumps({**base,'rows':rows}))
                with self.assertRaises(ValueError):ledger(path,{'seeds':FINAL})

    def test_truncated_journal_tail_is_not_a_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'progress.json';p.write_text(json.dumps(fixture_progress(0)))
            q={'seed':FINAL[0],'task':'touch','index':0}
            (root/'NEURAL_QUERIES.jsonl').write_text(json.dumps(q)+'\n{"incomplete":')
            check_pending_journal(p,{'seeds':FINAL})

    def test_journal_gap_duplicate_or_multiple_incomplete_tasks_rejected(self):
        for keys in ([('touch',0),('touch',2)],[('touch',0),('touch',0)],
            [('touch',0),('pick',0)]):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p=root/'progress.json';p.write_text(json.dumps(fixture_progress(0)))
                (root/'NEURAL_QUERIES.jsonl').write_text(''.join(json.dumps(
                    {'seed':FINAL[0],'task':task,'index':i})+'\n' for task,i in keys))
                with self.assertRaises(ValueError):check_pending_journal(p,{'seeds':FINAL})

    def test_s_callbacks_use_native_S_reference_and_restore(self):
        import VLA.autorefinement_r97.evaluate as evaluator
        import VLA.receipt_recovery_r88.map_reference as reference
        import VLA.receipt_recovery_r88.evaluate as bookkeeping
        from VLA.privileged_visual_r53.scene_map import render
        original=(evaluator.approved_final,reference.render,bookkeeping.render)
        with self.assertRaises(RuntimeError):
            with native_s_callbacks():
                self.assertIs(reference.render,render)
                self.assertIs(bookkeeping.render,render)
                self.assertIs(evaluator.approved_final,approved)
                raise RuntimeError('synthetic exception')
        self.assertEqual((evaluator.approved_final,reference.render,bookkeeping.render),original)

    def test_durable_commits_fsync_trace_and_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'NEURAL_QUERIES.jsonl').write_text('{}\n')
            with mock.patch('VLA.postfit800_r98.common.os.fsync',wraps=os.fsync) as sync:
                durable_json(root/'progress.json',{'completed_trials':0,'metadata':Path('safe')})
                self.assertEqual(sync.call_count,3)
                self.assertEqual(read(root/'progress.json')['metadata'],'safe')

    def test_query_durable_before_outcome_commit(self):
        from VLA.common.artifact_io import atomic_json as write
        sync=os.fsync;events=[]
        def observed_write(*args):
            events.append('write');return write(*args)
        def observed_sync(fd):
            events.append('sync');return sync(fd)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'NEURAL_QUERIES.jsonl').write_text('{}\n')
            with mock.patch('VLA.postfit800_r98.common.atomic_json',side_effect=observed_write), \
                mock.patch('VLA.postfit800_r98.common.os.fsync',side_effect=observed_sync):
                durable_json(root/'progress.json',{'synthetic':True})
            self.assertEqual(events,['sync','write','sync','sync'])

    def test_legacy_optimizer_is_never_signaled(self):
        from .owner import checked_signal
        ident={'pid':123,'command':['python','-m','VLA.recovery_quality_r90.train',
            '--family','vlanext_s','fit','--output',str(CP)]}
        with mock.patch('VLA.postfit800_r98.owner.alive',return_value=True), \
            mock.patch('VLA.postfit800_r98.owner.identity',return_value=ident), \
            mock.patch('VLA.postfit800_r98.owner.os.kill') as kill:
            for role in ('owned','legacy_scheduler'):
                with self.assertRaises(ValueError):checked_signal(ident,signal.SIGTERM,role)
            kill.assert_not_called()

    def test_pid_reuse_or_unrelated_process_never_signaled(self):
        from .owner import checked_signal
        ident={'pid':123,'command':['python','-m','VLA.postfit800_r98.evaluate']}
        with mock.patch('VLA.postfit800_r98.owner.alive',return_value=True), \
            mock.patch('VLA.postfit800_r98.owner.identity',return_value={**ident,'start_ticks':'changed'}), \
            mock.patch('VLA.postfit800_r98.owner.os.kill') as kill:
            with self.assertRaises(ValueError):checked_signal(ident,signal.SIGTERM,'owned')
            kill.assert_not_called()

    def test_official_signed_gripper_keeps_pose(self):
        from VLA.continuous_r86.vlanext_server import native_gripper
        a=np.arange(21,dtype=np.float32).reshape(3,7)/100
        a[:,6]=[-.00001,0.,.00001];b=native_gripper(a)
        np.testing.assert_array_equal(a[:,:6],b[:,:6])
        np.testing.assert_array_equal(b[:,6],[-1.,-1.,1.])

    def test_no_training_command_in_new_owner(self):
        from .owner import Owner
        owner=Owner()
        with mock.patch('VLA.postfit800_r98.owner.verify'):
            with self.assertRaises(ValueError):owner.spawn(Path('/python'),'VLA.recovery_quality_r90.train',
                ['--family','vlanext_s','fit','--output',CP],'must_not_launch')

    def test_partial_audit_is_explicit_not_default_complete(self):
        from .owner import audit_result
        with mock.patch('VLA.autorefinement_r97.audit.audit',return_value={'synthetic':True}) as audit:
            result=audit_result(Path('fixture'),require_complete=False)
            audit.assert_called_once_with(Path('fixture'),require_complete=False)
            self.assertEqual(result,{'synthetic':True})


class CrossAttemptChain(unittest.TestCase):
    """Two synthetic interrupted attempts, including committed failures."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='r98_synthetic_NOT_experiment_')
        self.root=Path(self.temp.name);self.rows=[];seed=FINAL[0]
        for index,tasks in enumerate((TASKS[:2],TASKS[2:])):
            folder=self.root/f'attempt{index}';source=folder/f'seed{seed}'/'trajectory.json'
            source.parent.mkdir(parents=True);traces={};queries=[]
            for task in tasks:
                traces[task]=[{'step':1,'success':False,'action':[0.]*7,
                    'query_sha256':'f'*64,'post_action_scene64':[0.]*64,'safety_violation':None}]
                self.rows.append({'seed':seed,'task':task,'success':False,'steps':1,
                    'trajectory':str(source),'safety_violation':None})
                queries.append({'seed':seed,'task':task,'index':0,'actions':[[0.]*7],
                    'input_sha256':'f'*64,'receipt':{'future_teacher_phase_or_labels':False,
                        'geometric_action_residual_or_Oracle':False}})
            source.write_text(json.dumps(traces))
            (folder/'NEURAL_QUERIES.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
        self.progress=self.root/'attempt1/progress.json'
        self.plan={'method':{'code':{},'weights_sha256':{},'checkpoint':'synthetic_fixture'},
            'cohort':{'role':'development_selection','seeds':[seed]}}
        (self.progress.parent/'PLAN.json').write_text(json.dumps(self.plan))
        self.save(4)

    def tearDown(self):self.temp.cleanup()

    def save(self,count):
        rows=self.rows[:count]
        self.progress.write_text(json.dumps({'rows':rows,'per_action':metrics(rows),
            'completed_trials':count,'planned_trials':4,'motor_controls':count,
            'status':'completed_actual_development' if count==4 else 'integration_error_fixture',
            'cohort_role':'development_selection'}))

    def test_all_failures_retained_with_complete_query_chain(self):
        from .owner import audit_result
        result=audit_result(self.progress)
        self.assertTrue(result['complete'])
        self.assertEqual(result['actual_neural_controls_verified'],4)
        self.assertEqual(sum(v['successes'] for v in result['per_action'].values()),0)

    def test_partial_chain_not_false_completion(self):
        from .owner import audit_result
        self.save(2)
        result=audit_result(self.progress,require_complete=False)
        self.assertFalse(result['complete'])
        self.assertEqual(result['actual_complete_trials_verified'],2)
        with self.assertRaises(ValueError):audit_result(self.progress)

    def test_replaced_model_action_rejected(self):
        from .owner import audit_result
        source=Path(self.rows[-1]['trajectory']);traces=read(source)
        traces['pick_place'][0]['action'][0]=.01;source.write_text(json.dumps(traces))
        with self.assertRaises(ValueError):audit_result(self.progress)


if __name__=='__main__':unittest.main()
