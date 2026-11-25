"""Mocked scheduler tests; these are NOT native GPU/model success evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from VLA.autocycle_r55 import run


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class SchedulerTests(unittest.TestCase):
    def exercise(self,broken=None):
        with tempfile.TemporaryDirectory(prefix='r55_scheduler_test_') as tmp:
            root=Path(tmp);output=root/'run'
            write(root/'preflight/audit.json',{'passed':True,'data_audit_sha256':'fake','code':{}})
            baselines={}
            for model in run.MODELS:
                rows=[{'seed':s,'task':t,'success':i<24,'safety_violation':None}
                    for i,(s,t) in enumerate((s,t) for s in range(20268011,20268021) for t in ('touch','pick','move','pick_place'))]
                p=root/(model+'_base.json');cp=root/(model+'_fake_checkpoint')
                write(p,{'completed_trials':40,'planned_trials':40,'successes':24,'rows':rows,'checkpoint':str(cp)})
                baselines[model]={'checkpoint':str(cp),'development_progress':str(p),'mode':'map3',
                    'prefix':5,'schedule':str(root/'unused_schedule.json'),'preserved_baseline':True}
            active={'checkpoint':None};final_models=[];servers=[];gpu_calls=[]
            class Process:
                pid=99999999
                def __init__(self,server=False,code=0):self.server=server;self.returncode=None if server else code
                def poll(self):return self.returncode
                def terminate(self):self.returncode=0
                def kill(self):self.returncode=-9
                def wait(self,timeout=None):return self.returncode
            def popen(command,**kwargs):
                module=command[command.index('-m')+1]
                def arg(name):return command[command.index(name)+1]
                if module.endswith('.server'):
                    active['checkpoint']=str(Path(arg('--checkpoint')).resolve());p=Process(True);servers.append(p);return p
                if module=='VLA.autocycle_r55.audit_video':return Process()
                self.assertEqual(module,'VLA.autocycle_r55.evaluate')
                model=arg('--model');cohort=json.loads(Path(arg('--cohort')).read_text())
                folder=Path(arg('--output'));role=cohort['role']
                if broken==model and role=='development_confirmation':return Process(code=1)
                if role=='untouched_final':
                    frozen=json.loads((output/'ALL_METHODS_FROZEN_BEFORE_FINAL.json').read_text())
                    self.assertEqual(len(frozen['methods']),4-(broken is not None))
                    self.assertFalse(frozen['subsequent_training_allowed']);final_models.append(model)
                n=4*len(cohort['seeds']);count=24 if n==40 else 72
                rows=[{'seed':s,'task':t,'success':i<count,'safety_violation':None}
                    for i,(s,t) in enumerate((s,t) for s in cohort['seeds'] for t in ('touch','pick','move','pick_place'))]
                write(folder/'progress.json',{'completed_trials':n,'planned_trials':n,'successes':count,'rows':rows,
                    'checkpoint':active['checkpoint']})
                return Process()
            def read(path):
                if Path(path)==run.R54/'status.json':
                    return {'status':'completed','active_child_pid':None,'active_server_pid':None}
                if str(path).endswith('final_geometry_exclusion.json') or str(path).endswith('final_cohort_exclusion_audit.json'):
                    return {'cases':[]}
                return json.loads(Path(path).read_text())
            with patch.object(run,'BASE',root),patch.object(run,'read',side_effect=read),\
                 patch.object(run,'hashes',return_value={}),patch.object(run,'verify'),\
                 patch.object(run,'sha256_file',return_value='fake'),\
                 patch.object(run,'initial_candidates',side_effect=lambda m:(baselines[m],None)),\
                 patch.object(run.subprocess,'Popen',side_effect=popen),\
                 patch.object(run.requests,'get',side_effect=lambda *args,**kwargs:SimpleNamespace(
                     raise_for_status=lambda:None,json=lambda:{'checkpoint':active['checkpoint']})),\
                 patch('VLA.new_vlas.train.wait_gpu',side_effect=lambda p:gpu_calls.append(p)),\
                 patch('VLA.run_feedback_refinement_r52.qualify_final_seeds',return_value={'cases':[]}):
                run.main(SimpleNamespace(output=output))
            state=json.loads((output/'status.json').read_text())
            self.assertEqual(state['status'],'completed')
            self.assertEqual(set(final_models),set(run.MODELS)-({broken} if broken else set()))
            self.assertTrue(all(p.poll() is not None for p in servers))
            self.assertEqual(set(state['errors']),{broken} if broken else set())
            report=json.loads((output/'report_results.json').read_text())
            self.assertEqual(report['all_four_complete'],broken is None)
            return state

    def test_four_models_freeze_before_any_final_and_release_servers(self):self.exercise()

    def test_one_native_integration_failure_does_not_fake_zero_or_stop_other_models(self):self.exercise('smolvla')


if __name__=='__main__':unittest.main()
