"""Reuse still-valid native input proofs; audit BOTH new TRAIN samplers."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.data import Data
from .common import DATA,R54_BASE,ROOT,hashes,read,verify
from .sampling import audit


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    reused=[]
    for model in ('smolvla','pi0'):
        p=R54_BASE/(model+'_input_final/audit.json');proof=read(p);verify(proof['code'])
        if not proof['passed'] or proof['samples']!=45 or proof['data_audit_sha256']!=sha256_file(DATA/'audit.json'):
            raise ValueError('Exact existing native45 observation proof no longer applies')
        reused.append({'model':model,'proof':str(p),'sha256':sha256_file(p),'actual_native_frames':45})
    for name in ('xvla_input_v3','adapter_input_v3'):
        p=ROOT/'VLA/audits/privileged_visual_r53_20261016'/name/'audit.json';proof=read(p)
        if not proof['passed'] or proof['data_audit_sha256']!=sha256_file(DATA/'audit.json'):
            raise ValueError('Exact native X/Adapter observation proof no longer applies')
        reused.append({'native_profile':name,'proof':str(p),'sha256':sha256_file(p)})
    data=Data(DATA)
    records={profile:audit(data,profile) for profile in ('acquisition','transport')}
    if min(r['counts']['XY_over5mm'] for r in records.values())<800:
        raise ValueError('Cannot pass the inherited native20percent XY gate')
    atomic_json(a.output/'audit.json',{'passed':True,'data':str(DATA),'data_audit_sha256':sha256_file(DATA/'audit.json'),
        'unchanged_native_observations_reused_proofs':reused,'new_TRAIN_sampler_audits':records,'code':hashes(),
        'actual_new_GPU_gates_still_required':True,'no_neural_SR_or_global_maximum_claim':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
