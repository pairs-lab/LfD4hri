"""Repair one missing metadata key without altering any native learned value."""
import argparse
import os
from pathlib import Path
import shutil

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import SCHEMA, read, verify, checkpoint_contract


def migrate(source, destination):
    source, destination = source.resolve(), destination.resolve()
    original = source / 'r54_contract.json'
    cfg = read(original)
    if (cfg.get('model') != 'pi0' or cfg.get('state_schema') != SCHEMA
        or cfg.get('schema') not in (None, SCHEMA)
        or cfg.get('action_schema') != 'native_EEF_delta_and_closure01_R54'
        or cfg.get('added_optimizer_updates') != 2
        or not cfg.get('native_VLA_actions_only')):
        raise ValueError('Only the known, actually trained native pi0 gate2 is migratable')
    verify(cfg['code'])
    if sha256_file(Path(cfg['data'])/'audit.json') != cfg['data_audit_sha256']:
        raise ValueError('Original qualified TRAIN receipt changed')
    proof = read(cfg['precision'])
    if (sha256_file(cfg['precision']) != cfg['precision_sha256'] or not proof.get('passed')
        or proof.get('checkpoint') != str(source) or proof.get('actual_optimizer_updates') != 2
        or proof.get('actual_microsteps') != 8 or proof.get('changed_trainable_tensors', 0) <= 0
        or not proof.get('saved_masters_and_moments_FP32')):
        raise ValueError('Actual saved precision/counter proof is missing or changed')
    before = sha256_file(original)
    destination.mkdir(parents=True, exist_ok=False)
    linked = []
    # Native Orbax payloads are immutable. Hardlink only these unchanged
    # payloads. Metadata is independently copied, NEVER overwritten through
    # a hardlink into the preserved source checkpoint.
    for item in source.iterdir():
        if item.name == 'r54_contract.json':
            continue
        if item.name in ('params', 'train_state', 'assets') and item.is_dir():
            shutil.copytree(item, destination/item.name, copy_function=os.link)
            linked.append(item.name)
        elif item.is_dir():
            shutil.copytree(item, destination/item.name)
        else:
            shutil.copy2(item, destination/item.name)
    fixed = {**cfg, 'schema': SCHEMA,
        'code': {**cfg['code'], str(Path(__file__).resolve()): sha256_file(__file__)},
        'r56_metadata_migration': {'original_checkpoint': str(source),
            'original_contract_sha256': before, 'new_optimizer_updates': 0,
            'inherited_actual_optimizer_updates': 2,
            'native_payloads_unchanged_hardlinked': linked,
            'original_saved_precision_proof_retained': True}}
    atomic_json(destination/'r54_contract.json', fixed)
    checkpoint_contract(destination)
    if sha256_file(original) != before:
        raise ValueError('Original metadata was modified')
    atomic_json(destination.parent/'migration_receipt.json', {
        'passed': True, 'checkpoint': str(destination),
        'source': str(source), 'source_contract_sha256': before,
        'repaired_contract_sha256': sha256_file(destination/'r54_contract.json'),
        'new_optimizer_updates': 0, 'native_reload_still_required': True})
    print(str(destination), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    migrate(a.source, a.destination)
