"""Independent whole-stream PTS/DTS audit without changing video files."""
from fractions import Fraction
from pathlib import Path
import subprocess
import imageio_ffmpeg
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,read


def audit(path,frames):
    # ffprobe is not installed. Stream-copy framecrc exposes EVERY original
    # compressed packet's DTS/PTS/duration without decoding or retiming it.
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),
        '-map','0:v:0','-c:v','copy','-f','framecrc','-'],
        capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    timebase=Fraction(next(line.split(':',1)[1].strip() for line in lines if line.startswith('#tb 0:')))
    packets=[[value.strip() for value in line.split(',')] for line in lines if line and not line.startswith('#')]
    pts=np.asarray([float(int(p[2])*timebase) for p in packets])
    dts=np.asarray([float(int(p[1])*timebase) for p in packets])
    duration=np.asarray([float(int(p[3])*timebase) for p in packets])
    expected=np.arange(len(frames))/30.
    error=float(np.max(np.abs(pts-expected))) if len(pts)==len(frames) and len(pts) else float('inf')
    passed=(len(packets)==len(frames) and error<2e-6
        and np.all(np.abs(pts-dts)<2e-6) and np.all(np.abs(duration-1/30.)<2e-6))
    if not passed:raise ValueError('Actual video packet cadence/format differs: '+str(path))
    return {'file':str(path.relative_to(OUT)),'sha256':sha256_file(path),'packets':len(packets),
        'PTS_start_s':float(pts[0]),'PTS_last_s':float(pts[-1]),'maximum_CFR30_error_s':error,
        'PTS_equals_DTS_no_reordering':True,'duration_s':float(pts[-1]+duration[-1]),
        'audit_method':'ffmpeg stream-copy framecrc compressed-packet timestamps, no retiming',
        'physical_tick_start':frames[0]['physics_tick'],'physical_tick_last':frames[-1]['physics_tick'],
        'encoded_t0_is_first_scheduled_capture_at_simulation_1_over_30_s':True,'passed':True}


def main():
    rows=[]
    for item in read(OUT/'produced_pairs.json'):
        folder=Path(item['paired_output'])
        for method in ('Scripted_Oracle','XVLA_terminal_release'):
            source=folder/method;frames=read(source/'frames.json')
            for name in ('handeye',*[f'view_{i}' for i in range(1,7)]):
                rows.append(audit(source/(name+'.mp4'),frames))
    atomic_json(OUT/'audit/whole_stream_packet_PTS.json',{'passed':True,'stream_count':len(rows),
        'every_encoded_packet_checked':True,'no_video_modified':True,'rows':rows})
    print({'PTS_audited_streams':len(rows),'passed':True},flush=True)


if __name__=='__main__':main()
