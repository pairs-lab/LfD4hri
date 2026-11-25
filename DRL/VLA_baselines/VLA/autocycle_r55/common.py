from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import ROOT,DATA,PRIOR,BASE as R54_BASE,SIM,PI,read,verify,hashes as r54_hashes

BASE=ROOT/'VLA/audits/autocycle_r55_20261017'
R54=R54_BASE/'automatic_four_family_v1'
MODELS=('xvla','vla_adapter','smolvla','pi0')
TASKS=('touch','pick','move','pick_place')


def hashes():
    result=r54_hashes()
    from VLA.privileged_visual_r53.contract import source_hashes
    result.update(source_hashes())
    result.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    return result


def optimizer_updates(checkpoint):
    p=Path(checkpoint)
    cfg=read(p/('config.json' if (p/'saved_values.json').exists() else 'r54_contract.json'))
    return cfg.get('r55_cumulative_optimizer_updates',cfg.get('actual_optimizer_updates',cfg.get('added_optimizer_updates')))
