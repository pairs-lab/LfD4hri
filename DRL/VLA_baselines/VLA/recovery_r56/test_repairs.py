"""CPU regressions for versioned repair boundaries, not a substitute for GPU gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import SCHEMA
from .entry import bind_contracts, main
from .migrate_pi0 import migrate


class Repairs(unittest.TestCase):
    def fixture(self, root):
        source=root/'old_gate2';source.mkdir()
        data=root/'data';data.mkdir();atomic_json(data/'audit.json',{'passed':True})
        for kind in ('params','train_state','assets'):
            folder=source/kind;folder.mkdir();(folder/'payload').write_bytes(b'original immutable payload')
        atomic_json(source/'scene32_checkpoint_contract.json',{'original_metadata':True})
        proof=root/'precision.json'
        atomic_json(proof,{'passed':True,'checkpoint':str(source),'actual_optimizer_updates':2,
            'actual_microsteps':8,'changed_trainable_tensors':30,'saved_masters_and_moments_FP32':True})
        atomic_json(source/'r54_contract.json',{'model':'pi0','state_schema':SCHEMA,
            'action_schema':'native_EEF_delta_and_closure01_R54','added_optimizer_updates':2,
            'native_VLA_actions_only':True,'code':{},'data':str(data),
            'data_audit_sha256':sha256_file(data/'audit.json'),
            'precision':str(proof),'precision_sha256':sha256_file(proof)})
        return source

    def test_migration_does_not_mutate_original_and_adds_no_training(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=self.fixture(root);target=root/'repaired/checkpoint'
            before={str(p.relative_to(source)):sha256_file(p) for p in source.rglob('*') if p.is_file()}
            migrate(source,target)
            after={str(p.relative_to(source)):sha256_file(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            self.assertEqual((source/'params/payload').stat().st_ino,(target/'params/payload').stat().st_ino)
            self.assertNotEqual((source/'scene32_checkpoint_contract.json').stat().st_ino,
                (target/'scene32_checkpoint_contract.json').stat().st_ino)
            fixed=json.loads((target/'r54_contract.json').read_text())
            self.assertEqual(fixed['schema'],SCHEMA)
            self.assertEqual(fixed['r56_metadata_migration']['new_optimizer_updates'],0)
            self.assertEqual(fixed['added_optimizer_updates'],2)
            with self.assertRaises(FileExistsError):migrate(source,target)

    def test_reject_different_action_schema(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=self.fixture(root);path=source/'r54_contract.json'
            value=json.loads(path.read_text());value['action_schema']='joint_not_EEF'
            atomic_json(path,value)
            with self.assertRaises(ValueError):migrate(source,root/'repaired')

    def test_reject_changed_actual_precision_proof(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=self.fixture(root)
            atomic_json(root/'precision.json',{'passed':True,'actual_optimizer_updates':2000})
            with self.assertRaises(ValueError):migrate(source,root/'repaired')

    def test_new_pi_contract_has_correct_discriminator_without_changing_training_schema(self):
        calls=[];module=SimpleNamespace(atomic_json=lambda p,v:calls.append((p,v)))
        bind_contracts(module)
        module.atomic_json('checkpoint/r54_contract.json',{'state_schema':SCHEMA,
            'native_VLA_actions_only':True,'code':{},'added_optimizer_updates':1500})
        value=calls[0][1]
        self.assertEqual(value['schema'],SCHEMA)
        self.assertEqual(value['state_schema'],SCHEMA)
        self.assertEqual(value['added_optimizer_updates'],1500)
        self.assertIn(str(Path(__file__).parent/'entry.py'),value['code'])

    def test_smol_registration_happens_before_native_trainer_import(self):
        calls=[]
        def imported(name):
            self.assertEqual(calls,['registered'])
            return SimpleNamespace(main=lambda args:calls.append(args),atomic_json=lambda *_:None)
        with patch('sys.argv',['entry','fit_smol','--mode','gate','--updates','2','--output','out']), \
             patch('VLA.recovery_r56.entry.register_smol',side_effect=lambda:calls.append('registered')), \
             patch('VLA.recovery_r56.entry.importlib.import_module',side_effect=imported):
            main()
        self.assertEqual(calls[1].updates,2)
        self.assertEqual(calls[1].mode,'gate')


if __name__=='__main__':
    unittest.main()
