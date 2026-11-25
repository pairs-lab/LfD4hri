"""Automatically extract, independently verify, and prepare review artifacts."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .export import OUTPUT,ROOT,THREAD_ENV,FFMPEG,read,write,sha


def main(out):
    if not read(out/'PILOT_AND_ADMISSION.json')['passed']:raise ValueError('Full extraction requires passed pilot')
    with (out/'.automatic.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).parent
        write(out/'EXTRACTION_SOURCE_MANIFEST.json',{'sources':{str(p):sha(p) for p in source.glob('*.py')},
            'ffmpeg_binary':FFMPEG,'ffmpeg_binary_sha256':sha(FFMPEG),'python':sys.executable,
            'CPU_only_single_worker':True,'no_robot_environment_or_model_imports':True})
        stages=[('unit_tests',['-m','VLA.frame_export_r61.tests','-v']),
                ('extract',['-m','VLA.frame_export_r61.export','run','--output',str(out)]),
                ('independent_final_verify',['-m','VLA.frame_export_r61.export','verify','--output',str(out)]),
                ('prepare_actual_frame_contact_sheets',['-m','VLA.frame_export_r61.review','--output',str(out)])]
        for label,args in stages:
            log=out/(label+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=THREAD_ENV,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                write(out/'AUTOMATION.json',{'status':'running','stage':label,'pid':os.getpid(),'child':child.pid,
                    'UTC':datetime.now(timezone.utc).isoformat()})
                code=child.wait()
            if code:
                write(out/'AUTOMATION.json',{'status':'stopped_on_real_error_outputs_preserved','stage':label,
                    'exit_code':code,'log':str(log),'UTC':datetime.now(timezone.utc).isoformat()})
                raise RuntimeError('Stage failed: '+label+'; see '+str(log))
        write(out/'AUTOMATION.json',{'status':'all_frames_ready_actual_visual_review_required',
            'UTC':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    main(p.parse_args().output)
