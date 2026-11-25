"""Synthetic report fixtures only; no simulator, GPU or model execution."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.common.artifact_io import atomic_json
from .report import TASKS,per_action,forecast_for_task,refresh


def forecast(task,rate):
    return {'label':'FORECAST_ONLY_fixture','actual_additional_cohorts_run':0,
        'actual_evaluation_cohorts':1,'training_seed_variance_estimable':False,
        'by_task':{task:{'measured_one_cohort_SR':rate,
            'forecast_five_cohort_mean_median':rate,
            'forecast_five_cohort_sample_SD_median':.02,
            'forecast_five_cohort_mean_95_interval':[max(0,rate-.1),min(1,rate+.1)],
            'forecast_five_cohort_sample_SD_95_interval':[.01,.04]}}}


class ReportTests(unittest.TestCase):
    def test_action_denominators_are_separate_missing_is_not_zero_SR(self):
        rows=[{'seed':1,'task':'touch','success':True},{'seed':2,'task':'touch','success':False},
              {'seed':1,'task':'pick','success':False}]
        result=per_action(rows,[1,2],200)
        self.assertEqual(result['touch']['measured_SR'],.5)
        self.assertEqual(result['pick']['measured_SR'],0)
        self.assertIsNone(result['move']['measured_SR'])
        self.assertEqual(result['touch']['completed'],2)
        self.assertEqual(result['pick']['completed'],1)

    def test_duplicates_unknown_tasks_and_nonboolean_outcomes_are_rejected(self):
        v={'seed':1,'task':'touch','success':True}
        for rows in ([v,v],[{**v,'task':'unknown'}],[{**v,'success':1}],[{**v,'seed':3}]):
            with self.assertRaises(ValueError):per_action(rows,[1,2],200)

    def test_no_forecast_until_whole_cohort_audited(self):
        f=forecast('touch',.75)
        self.assertIsNone(forecast_for_task(f,'touch',{'completed':199,'measured_SR':.75},True))
        self.assertIsNone(forecast_for_task(f,'touch',{'completed':200,'measured_SR':.75},False))

    def test_forecast_percentage_points_and_explicit_unmeasured_label(self):
        v=forecast_for_task(forecast('touch',.75),'touch',{'completed':200,'measured_SR':.75},True)
        self.assertEqual(v['mean_percent'],75)
        self.assertEqual(v['sample_SD_percentage_points'],2)
        self.assertEqual(v['actual_additional_cohorts_run'],0)
        self.assertFalse(v['training_seed_variance_measured'])

    def test_wrong_forecast_or_training_seed_claim_is_rejected(self):
        f=forecast('touch',.75)
        with self.assertRaises(ValueError):forecast_for_task(f,'touch',{'completed':200,'measured_SR':.8},True)
        f['training_seed_variance_estimable']=True
        with self.assertRaises(ValueError):forecast_for_task(f,'touch',{'completed':200,'measured_SR':.75},True)

    def test_full_fixture_has_four_pages_and_no_cross_action_aggregate(self):
        with tempfile.TemporaryDirectory(prefix='r63_SYNTHETIC_report_fixture_') as name:
            root=Path(name);base=root/'base';old=root/'r59';out=root/'report'
            seeds=list(range(200));rows=[{'seed':s,'task':t,'success':True} for s in seeds for t in TASKS]
            atomic_json(base/'PREDECLARATION.json',{'seeds':seeds})
            atomic_json(base/'owner_status.json',{'status':'running','stage':'synthetic_unit_fixture'})
            atomic_json(old/'episodes/progress.json',{'rows':rows,'status':'completed'})
            atomic_json(old/'FINAL_AUDIT.json',{'passed':True})
            atomic_json(base/'xvla/RESULTS.json',{'actual_completed_trials':800,
                'statistics':{'by_task':{t:{'successes':200,'trials':200} for t in TASKS}}})
            f=forecast('touch',1.)
            f['by_task']={t:forecast(t,1.)['by_task'][t] for t in TASKS}
            atomic_json(base/'xvla/FORECAST_NOT_MEASURED_5_SEEDS.json',f)
            with patch('VLA.action_reports_r63.report.R59',old),patch('VLA.action_reports_r63.report.R56',root/'absent'):
                refresh(base,out)
            d=json.loads((out/'PER_ACTION.json').read_text())
            self.assertEqual(set(d['by_action']),{'touch','pick','move','put'})
            self.assertNotIn('overall',json.dumps(d))
            for t in ('touch','pick','move','put'):self.assertTrue((out/(t+'.md')).is_file())
            with (out/'MEASURED_BY_ACTION.csv').open() as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),16)
            self.assertTrue(all(r['measured_SR_percent']=='' for r in table if r['model']!='xvla'))
            with (out/'FORECAST_BY_ACTION.csv').open() as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),4)
            self.assertTrue(all(r['model']=='xvla' for r in table))


if __name__=='__main__':unittest.main()
