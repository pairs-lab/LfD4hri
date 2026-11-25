"""CPU-only contract/fault-injection tests. Synthetic outcomes stay in /tmp only."""
import copy
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock
from VLA.autonomy_r83.common import metrics
from VLA.common.artifact_io import atomic_json
from . import common,guardian
from .audit import audit_chain,certify
from .evaluate import incomplete_prefix

REAL=common.ROOT/'VLA/audits/dita800_user_r87_20261021/final800/progress.json'


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dita_guardian_contract_')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def ledger(self,n=2):
        """Deliberately synthetic, isolated fixtures, NOT experiment observations."""
        folder=self.root/'synthetic_not_experimental/final800';folder.mkdir(parents=True)
        rows=[];queries=[]
        for seed,task in common.ORDERED[:n]:
            tracefile=folder/f'seed{seed}'/'trajectory.json'
            traces=common.read(tracefile) if tracefile.exists() else {}
            digest=f'SYNTHETIC_TEST_ONLY_{seed}_{task}'
            traces[task]=[{'step':1,'action':[0.]*7,'query_sha256':digest,
                'post_action_scene64':[0.]*64,'success':False}]
            atomic_json(tracefile,traces)
            rows.append({'seed':seed,'task':task,'success':False,'steps':1,'trajectory':str(tracefile)})
            queries.append({'seed':seed,'task':task,'index':0,'input_sha256':digest,'actions':[[0.]*7],
                'receipt':{'future_teacher_phase_or_labels':False,'geometric_action_residual_or_Oracle':False}})
        (folder/'NEURAL_QUERIES.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
        data={'rows':rows,'completed_trials':n,'planned_trials':800,'motor_controls':n,'native_queries':n,
            'per_action':metrics(rows),'status':'completed_actual_final800' if n==800 else 'running_actual_neural'}
        atomic_json(folder/'progress.json',data)
        method={'frozen_after_confirmation':False,'frozen_for_user_requested_benchmark':True,
            'benchmark_authorization':{'user_requested_800_despite_failed_pilot_gate':True},
            'code':{},'weights_sha256':{}}
        atomic_json(folder/'PLAN.json',{'method':method,
            'cohort':{'role':'fresh_secondary_final800','seeds':list(range(20284001,20284201))}})
        return folder/'progress.json'
    def test_real_committed243_kept_partial174_not_counted(self):
        data,rows,keys=common.complete_rows(REAL)
        self.assertEqual(len(rows),243);self.assertEqual(len(keys),243)
        self.assertEqual(data['motor_controls']-sum(r['steps'] for r in rows),174)
    def test_duplicate_out_of_cohort_and_nonchronological_ledger_rejected(self):
        p=self.ledger();original=common.read(p)
        cases=[]
        d=copy.deepcopy(original);d['rows'][1]=d['rows'][0];cases.append(d)
        d=copy.deepcopy(original);d['rows'][0]['seed']=9999;cases.append(d)
        d=copy.deepcopy(original);d['rows'].reverse();cases.append(d)
        for data in cases:
            atomic_json(p,data)
            with self.assertRaises(ValueError):common.complete_rows(p)
    def test_real_process_identity_and_unrelated_python_not_signalable(self):
        import os
        ident=common.identity(os.getpid())
        self.assertIsNotNone(ident);self.assertTrue(common.alive(os.getpid()))
        with patch.object(guardian.os,'kill') as kill:
            with self.assertRaises(ValueError):guardian.checked_signal(ident,signal.SIGTERM)
            kill.assert_not_called()
    def test_pid_reuse_and_disappearing_process_never_signalled(self):
        ident={'pid':42,'start_ticks':'1','command':['python','-m','VLA.receipt_recovery_r88.run']}
        with patch.object(guardian,'identity',return_value={**ident,'start_ticks':'2'}),\
             patch.object(guardian,'alive',return_value=True),patch.object(guardian.os,'kill') as kill:
            with self.assertRaises(ValueError):guardian.checked_signal(ident,signal.SIGTERM)
            kill.assert_not_called()
        with patch.object(guardian,'identity',return_value=None),patch.object(guardian.os,'kill') as kill:
            self.assertFalse(guardian.checked_signal(ident,signal.SIGTERM));kill.assert_not_called()
    def test_identity_cannot_silently_be_replaced(self):
        with self.assertRaises(ValueError):guardian.merge_identities({'pid':{'pid':1}}, {'pid':{'pid':2}})
    def test_transient_retry_but_integrity_and_deterministic_errors_fail_closed(self):
        p=self.root/'ERROR.json';p.write_text('ReadTimeout: local actual-neural server')
        self.assertTrue(guardian.retryable(self.root))
        p.write_text('ValueError: Incorrect actual model/input receipt')
        self.assertFalse(guardian.retryable(self.root,owner_missing=True,stalled=True))
        p.write_text('ReadTimeout');(self.root/'PREFIX_DIVERGENCE.json').write_text('{}')
        self.assertFalse(guardian.retryable(self.root,owner_missing=True))
    def test_live_optimizer_excludes_queue_pause(self):
        p=self.root/'LIVE.json';atomic_json(p,{'pid':42,'child_pid':None,'server_pid':None,'resource_wait':{'waiting':True}})
        ident={'pid':42,'command':['python','-m','VLA.continuous_r86.owner']}
        with patch.object(guardian,'identity',return_value=ident),\
             patch.object(guardian,'live_children',return_value={43}),patch.object(guardian,'checked_signal') as sig:
            self.assertIsNone(guardian.hold_idle_queue(p));sig.assert_not_called()
    def test_incomplete_query_prefix_ignores_uncommitted_tail_but_rejects_duplicates(self):
        j=self.root/'journal.jsonl';q={'seed':20284001,'task':'touch','index':0}
        j.write_text(json.dumps(q)+'\n'+ '{"seed":')
        self.assertEqual(len(incomplete_prefix(j,[20284001],{})),1)
        j.write_text((json.dumps(q)+'\n')*2)
        with self.assertRaises(ValueError):incomplete_prefix(j,[20284001],{})
        q['index']=1;j.write_text(json.dumps(q)+'\n')
        with self.assertRaises(ValueError):incomplete_prefix(j,[20284001],{})
    def test_physical_loop_dependencies_are_unchanged(self):
        from VLA.receipt_recovery_r88.evaluate import compile_resume
        from VLA.autonomy_r83 import evaluate as old
        fn=compile_resume(old,lambda:([],0,0),lambda c:[],lambda *a:False,
                          SimpleNamespace(Session=None),lambda *a:None)
        for name in ('model_input','environment_class_for_profile','randomize_rl_scene','select_tasks'):
            self.assertIs(fn.__globals__[name],getattr(old,name))
    def test_partial_never_certified_and_action_substitution_detected(self):
        p=self.ledger();chain=audit_chain(p)
        self.assertEqual(chain['actual_complete_trials_verified'],2)
        self.assertEqual(chain['actual_neural_controls_verified'],2)
        with self.assertRaises(ValueError):certify(p)
        row=common.read(p)['rows'][0];t=Path(row['trajectory']);d=common.read(t)
        d[row['task']][0]['action'][0]=.123;atomic_json(t,d)
        with self.assertRaisesRegex(ValueError,'replaced'):audit_chain(p)
    def test_all800_required_with200_each_and_no_false_confirmation_claim(self):
        p=self.ledger(800);result=certify(p)
        self.assertEqual(result['actual_trials'],800)
        self.assertTrue(all(v['n']==200 for v in result['per_action'].values()))
        self.assertFalse(result['confirmation_gate_passed'])
        self.assertEqual(result['actual_neural_controls_verified'],800)
    def test_snapshot_reuses_exact_committed_prefix_and_source_hash(self):
        p=self.ledger();out=self.root/'snapshots'
        with patch.object(guardian,'OUT',out):s=guardian.snapshot(p.parent.parent,1)
        saved=common.read(s)
        self.assertEqual(saved['rows'],common.read(p)['rows'])
        self.assertEqual(saved['original_progress_path'],str(p))
        self.assertEqual(saved['original_progress_sha256'],common.sha256_file(p))
    def test_oom_recovery_dispatch_without_real_process_or_gpu_changes(self):
        """Fault injection executes the actual guardian branch, mocks only OS/GPU."""
        p=self.ledger();current=p.parent.parent
        atomic_json(current/'status.json',{'pid':42,'status':'integration_error_NOT_model_SR'})
        (current/'ERROR.json').write_text('RuntimeError: CUDA out of memory')
        out=self.root/'guardian';ident={'pid':42,'start_ticks':'1',
            'command':['python','-m','VLA.receipt_recovery_r88.run']}
        atomic_json(out/'PREFLIGHT.json',{'passed':True,'code':common.bindings()})
        launched=MagicMock();launched.pid=43;launched.poll.return_value=None
        with patch.object(guardian,'CURRENT',current),patch.object(guardian,'OUT',out),\
             patch.object(guardian,'capture',return_value={'pid':ident}),\
             patch.object(guardian,'identity',return_value={'pid':43,'start_ticks':'2',
                 'command':['python','-m','VLA.benchmark_guardian_r89.attempt']}),\
             patch.object(guardian,'alive',return_value=True),patch.object(guardian,'hold_idle_queue',return_value=None),\
             patch.object(guardian,'stop_captured') as stop,\
             patch.object(guardian.subprocess,'Popen',return_value=launched) as launch,\
             patch.object(guardian.time,'sleep',side_effect=RuntimeError('END_SYNTHETIC_TEST')):
            with self.assertRaisesRegex(RuntimeError,'END_SYNTHETIC_TEST'):guardian.main()
            stop.assert_called_once();launch.assert_called_once()
            command=launch.call_args.args[0]
            self.assertIn('VLA.benchmark_guardian_r89.attempt',command)
            self.assertIn('--resume-progress',command)
        s=common.read(out/'resume_snapshot_01.json')
        self.assertEqual(s['completed_trials'],2)
        self.assertFalse((out/'CERTIFIED_RESULT.json').exists())


if __name__=='__main__':unittest.main()
