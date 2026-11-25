import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter
import numpy as np
from .evaluate import json_metadata,atomic_json
from .dita_train import balanced_gate
from .common import OUT
from VLA.common.artifact_io import atomic_json as evidence

class JsonContracts(unittest.TestCase):
    def test_nested_paths(self):
        obj={'protocol':Path('/a/protocol.json'),'nested':[{'model':Path('/b/model')}], 'ordinary':True}
        actual=json_metadata(obj)
        self.assertEqual(json.loads(json.dumps(actual)),{'protocol':'/a/protocol.json','nested':[{'model':'/b/model'}],'ordinary':True})
        self.assertIsInstance(obj['protocol'],Path)
    def test_actual_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'PLAN.json';atomic_json(p,{'protocol':Path('/p')})
            self.assertEqual(json.loads(p.read_text()),{'protocol':'/p'})
    def test_nan_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):atomic_json(Path(tmp)/'x.json',{'loss':float('nan')})

def main():
    from VLA.replacement_r84.data import Data
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(JsonContracts))
    if not result.wasSuccessful():raise ValueError('JSON repair unit failure')
    data=Data();rng=np.random.default_rng(2026842)
    old=[data.draw(rng) for _ in range(8)]
    fixed=[balanced_gate(data,i) for i in range(8)]
    old_counts=Counter(data.rows[ei]['task'] for ei,fi in old)
    counts=Counter(data.rows[ei]['task'] for ei,fi in fixed)
    if len(old_counts)!=3 or old_counts.get('pick_place',0)!=0:
        raise ValueError('Did not reproduce actual first GPU gate sampling error')
    if set(counts)!=set(('touch','pick','move','pick_place')) or set(counts.values())!={2}:
        raise ValueError('Corrected gate must include two actual anchors per action')
    evidence(OUT/'REPAIR_TESTS.json',{'passed':True,'JSON_tests':3,'old_exact8_draw_counts':dict(old_counts),
        'corrected_gate8_counts':dict(counts),'actual_TRAIN_anchors':fixed,
        'model_or_gradient_acceptance_criterion_changed':False,'robot_SR_measured':False})

if __name__=='__main__':main()
