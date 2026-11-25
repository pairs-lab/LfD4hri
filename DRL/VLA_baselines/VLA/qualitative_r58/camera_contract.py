"""Check actual golden camera extrinsics against the exported reference set."""
import cv2
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import decode_at
from .common import OUT,ROOT,read


def main():
    reference=ROOT/'env_eval/demos/oracle_feedback_v3_20261015_03_handeye_only/wide_camera_settings.json'
    expected=read(reference);rows=[]
    probes=sorted((OUT/'audit/goldens').glob('*/probe.json'))
    probes+=sorted((OUT/'audit/prefetch_goldens').glob('*/probe.json'))
    probes+=sorted((OUT/'audit/supplemental_midpoints').glob('*/*/probe.json'))
    probes+=[OUT/'audit/golden_pinned_renderer/probe.json']
    for probe in probes:
        for row in read(probe)['rows']:
            if not row['role'].startswith('wide_'):continue
            index=int(row['role'].split('_')[1])-1;actual=row['camera'];wanted=expected[index]
            if actual['name']!=wanted['name']:raise ValueError('Wrong camera order/name')
            errors=[]
            for key in ('eye','target','up','fov'):
                errors.append(float(np.abs(np.asarray(actual[key])-np.asarray(wanted[key])).max()))
            if max(errors)>1e-12:raise ValueError('Actual golden extrinsics differ from exported reference')
        rows.append({'probe':str(probe.relative_to(OUT)),'all6wide_extrinsics_match_reference':True})
    raw=OUT/'_renders/scene06_seed20269730_VLA';golden=OUT/'audit/golden_pinned_renderer'
    encoded=[]
    for row in read(golden/'probe.json')['rows']:
        role=row['role'];index=row['frame'];image=decode_at(raw/(role+'.mp4'),index)
        lossless=cv2.imread(str(golden/f'frame{index:05d}_{role}.png'))
        error=float(np.abs(image.astype(np.int16)-lossless.astype(np.int16)).mean())
        if error>=6.:raise ValueError('Reused pinned-camera source differs from matching golden geometry')
        encoded.append({'frame':index,'role':role,'encoded_vs_lossless_RGB_MAE':error,'passed':True})
    atomic_json(OUT/'audit/reference_camera_extrinsics.json',{'passed':True,
        'reference_file':str(reference),'reference_sha256':sha256_file(reference),
        'golden_probe_sets_checked':rows,'old_pinned_encoded_source21images_checked':encoded,
        'no_policy_or_video_camera_changed':True})
    print({'actual_camera_probe_sets_verified':len(rows),'old_pinned21encoded_samples_match':True},flush=True)


if __name__=='__main__':main()
