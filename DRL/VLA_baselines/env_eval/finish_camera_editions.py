"""Compose completed camera sources in parallel with ongoing render workers."""
import argparse,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
from env_eval.camera_editions import compose_scene,SETTINGS
from env_eval.oracle_demo import scenes


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output-parent',type=Path,default=Path('env_eval/demos'))
    p.add_argument('--name',default='oracle_feedback_v3_20261015');args=p.parse_args()
    settings=json.loads(SETTINGS.read_text());cv2.setNumThreads(1)
    def worker(label):
        for scene in scenes():
            relative=Path(label)/scene.action/f'config{scene.config}';source=args.cache/relative
            deadline=time.monotonic()+7200
            while not (source/'audit.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(str(source))
                time.sleep(2)
            for mode in ('front_handeye','handeye'):
                target=args.output_parent/(args.name+settings['folder_suffixes'][mode])/relative
                compose_scene(source,target,mode,settings)
                print('AUDITED',mode,relative,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(worker,['UF850','UR5e_Robotiq2F']):pass
    print('DONE 64 scene-editions, 480 per-configuration films',flush=True)


if __name__=='__main__':main()
