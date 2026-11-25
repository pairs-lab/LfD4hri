"""Remove duplicate media storage ONLY, retaining every pathname and byte.

No unique videos, weights, datasets or checkpoints are deleted. Same-filesystem
byte-identical redundant files become hardlinks; receipts preserve both paths.
Default is inventory only. --apply requires the user's cleanup authorization.
"""
import argparse
from collections import defaultdict
import filecmp
import os
from pathlib import Path
import time

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT,OUT


def opened_paths():
    result=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:result.add(fd.resolve())
                except (OSError,RuntimeError):pass
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    return result


def main(a):
    roots=[ROOT/'env_eval/demos',ROOT/'VLA/qualitative/ur5e_smolvla_handeye_r51_20261016']
    groups=defaultdict(list);active=opened_paths()
    reference='oracle_feedback_v3_20261015_'
    for root in roots:
        for p in root.rglob('*.mp4'):
            if p.is_symlink() or not p.is_file() or p.resolve() in active:continue
            s=p.stat()
            if s.st_size>=1024**2:groups[(s.st_dev,s.st_size)].append(p)
    pairs=[];released=0;hashed=0
    for candidates in groups.values():
        if len(candidates)<2:continue
        seen={}
        # Published reference files are preferred retained masters, never the
        # replacement/deletion target. Every existing path stays available.
        candidates.sort(key=lambda p:(reference not in str(p),str(p)))
        for p in candidates:
            before=p.stat();digest=sha256_file(p);hashed+=1
            time.sleep(.01)
            if digest not in seen:seen[digest]=p;continue
            master=seen[digest];s=master.stat()
            if (before.st_ino==s.st_ino or before.st_nlink!=1 or reference in str(p)):
                continue
            if p.resolve() in opened_paths() or not filecmp.cmp(master,p,shallow=False):continue
            row={'retained_master':str(master),'retained_path':str(p),'sha256':digest,
                'bytes':before.st_size,'reason':'byte-identical redundant MP4 storage; pathname retained as hardlink',
                'unique_content_deleted':False,'original_payload_recoverable_from_retained_master':True}
            if a.apply:
                current=p.stat()
                if (current.st_ino,current.st_mtime_ns,current.st_size)!=(before.st_ino,before.st_mtime_ns,before.st_size):
                    raise ValueError('Candidate changed during cleanup')
                temp=p.with_name(p.name+'.r57_verified_duplicate_link')
                os.link(master,temp)
                os.replace(temp,p)
                if sha256_file(p)!=digest:raise ValueError('Dedup did not preserve content')
                released+=before.st_blocks*512
                row['replaced_duplicate_inode_with_identical_hardlink']=True
            pairs.append(row)
            atomic_json(a.output,{'status':'running','apply':a.apply,'hashed_files':hashed,
                'reclaimed_allocated_bytes':released,'rows':pairs})
    atomic_json(a.output,{'status':'completed','apply':a.apply,'hashed_files':hashed,
        'reclaimed_allocated_bytes':released,'duplicate_files':len(pairs),'rows':pairs,
        'unique_content_deleted':False,'all_video_paths_and_SHA256_preserved':True,
        'weights_datasets_checkpoints_and_current_training_untouched':True,
        'retired_CogACT_weights_NOT_removed_because_old_live_queue_still_references_family':True})
    print({'duplicate_files':len(pairs),'reclaimed_allocated_bytes':released,'unique_content_deleted':False},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--output',type=Path,default=OUT/'cleanup_manifest.json')
    main(p.parse_args())
