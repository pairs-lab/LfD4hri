"""Immutable, clearly labelled WARM START migration, not a newly fitted VLA.

Only a separately qualified TRAIN normalizer/data manifest changes. Actual
weights, Adam and RNG are byte-identical links to parent. Must run2 real new
updates, fresh-process reload gate, then the declared long fit before testing.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.stable_native_r44.conditioning import add_contract, require_contract


def main(args):
    parent, out, data = args.parent.resolve(), args.output.resolve(), args.data.resolve()
    cfg = copy.deepcopy(require_contract(parent))
    audit = json.loads((data/'audit.json').read_text())
    if not audit.get('passed') or not audit.get('normalizer_schema_changed_EXPLICITLY_new_checkpoint_required'):
        raise ValueError('Qualified new TRAIN source required')
    out.mkdir(parents=True, exist_ok=False)
    saved = json.loads((parent/'saved_values.json').read_text())
    for name, field in [('trainable_masters.pt', 'trainable_master_sha256'), ('optimizer.pt', 'optimizer_sha256')]:
        if sha256_file(parent/name) != saved[field]:
            raise ValueError('Parent weights or Adam changed')
        os.link(parent/name, out/name)
    os.link(parent/'torch_rng.pt', out/'torch_rng.pt')
    old_updates = cfg['actual_optimizer_updates']
    old_stats = cfg['stats']
    cfg.update(stats=json.loads((data/'stats.json').read_text()), data_root=str(data),
        data_audit_sha256=sha256_file(data/'audit.json'), own_full_source_numeric_gate_not_final_model_SR=True,
        WARM_START_ONLY_DO_NOT_SERVE_OR_REPORT_AS_NEW_FIT=True,
        r52_lineage={'parent_checkpoint': str(parent), 'parent_actual_optimizer_updates': old_updates,
            'actual_additional_updates_on_recovery_data': 0,
            'all_weights_Adam_RNG_byte_identical_to_parent': True,
            'normalizers_changed_NEW_optimization_and_reload_required': True,
            'parent_statistics_sha256': sha256_file(parent/'config.json')})
    cfg['code'][str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    atomic_json(out/'parent_stats_NOT_new_stats.json', old_stats)
    atomic_json(out/'config.json', cfg)
    saved.update(config_sha256=sha256_file(out/'config.json'), reference_prediction_sha256=None,
        shape=None, finite_prediction=None, WARM_START_ONLY_new_normalizer_reference_NOT_yet_computed=True)
    atomic_json(out/'saved_values.json', saved)
    add_contract(out, parent=parent)
    atomic_json(out/'migration.json', {'parent': str(parent), 'new_training_data': str(data),
        'model_actual_updates_ADDED': 0, 'checkpoint_servable': False,
        'actions_or_physics_modified': False, 'next_required': 'gate2_then_fresh_reload_then_long_fit'})
    print(json.dumps({'warm_start': str(out), 'parent_updates': old_updates, 'new_updates': 0}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
