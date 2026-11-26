import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from protocol21.checkpoint import save,load
from protocol21.tests.test_inputs import stats


class CheckpointTests(unittest.TestCase):
    def test_native_bfloat16_roundtrip_not_lossily_promoted(self):
        with tempfile.TemporaryDirectory() as d:
            r=SimpleNamespace(model=torch.nn.Linear(21,4).bfloat16());p=Path(d)/'cp'
            old=r.model.weight.detach().clone();save(r,'vlanext_s',stats(),p)
            with torch.no_grad():r.model.weight.zero_()
            load(r,'vlanext_s',p,stats())
            self.assertEqual(r.model.weight.dtype,torch.bfloat16)
            torch.testing.assert_close(r.model.weight,old,rtol=0,atol=0)

    def test_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            r=SimpleNamespace(model=torch.nn.Linear(21,4));p=Path(d)/'cp'
            old={n:v.detach().clone() for n,v in r.model.named_parameters()}
            save(r,'flower',stats(),p)
            with torch.no_grad():r.model.weight.add_(1)
            load(r,'flower',p,stats())
            for n,v in r.model.named_parameters():torch.testing.assert_close(v,old[n],rtol=0,atol=0)

    def test_wrong_schema_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            r=SimpleNamespace(model=torch.nn.Linear(21,4));p=Path(d)/'cp';save(r,'flower',stats(),p)
            m=json.loads((p/'metadata.json').read_text());m['robot_state_dim']=67
            (p/'metadata.json').write_text(json.dumps(m))
            old=r.model.weight.detach().clone()
            with self.assertRaises(ValueError):load(r,'flower',p,stats())
            torch.testing.assert_close(old,r.model.weight,rtol=0,atol=0)
