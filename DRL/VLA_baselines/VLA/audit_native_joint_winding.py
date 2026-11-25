"""Full TRAIN audit for artificial angular jumps in OpenPI DeltaActions.

Source arrays are never edited. Reports both native representations and proves
same periodic FK input, before normalization/FAST codec and new fine-tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import nearest_joint_targets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--horizon",type=int,default=10);a=p.parse_args()
    if a.output.exists():raise FileExistsError("Keep previous diagnostics")
    rows=json.loads((a.source/"index.json").read_text());report=[];examples=[];periodic_error=0.;total=0
    for row in rows:
        with np.load(a.source/row["file"],allow_pickle=False) as data:
            q=data["joint_positions"].astype(np.float64);target=data["next_joint_targets"].astype(np.float64)
        n=len(q);ids=np.minimum(np.arange(n)[:,None]+np.arange(a.horizon),n-1)
        original=(target[ids]+np.pi)%(2*np.pi)-np.pi;current=(q+np.pi)%(2*np.pi)-np.pi
        delta=original-current[:,None,:]
        new=np.asarray([nearest_joint_targets(target[id],observed) for id,observed in zip(ids,q,strict=True)])
        lifted_delta=new-current[:,None,:]
        error=float(max(np.max(np.abs(np.sin(original)-np.sin(new))),np.max(np.abs(np.cos(original)-np.cos(new)))))
        periodic_error=max(periodic_error,error)
        if error>1e-6 or np.max(np.abs(lifted_delta))>np.pi+1e-6:raise ValueError("Physical winding equivalence failed")
        count=int(np.any(np.abs(delta)>np.pi,axis=(1,2)).sum())
        if count and len(examples)<12:
            frame=int(np.flatnonzero(np.any(np.abs(delta)>np.pi,axis=(1,2)))[0]);examples.append({"file":row["file"],
                "frame":frame,"old_delta_max_rad":float(np.max(np.abs(delta[frame]))),
                "nearest_delta_max_rad":float(np.max(np.abs(lifted_delta[frame])))})
        report.append({"file":row["file"],"frames":n,"frames_with_artificial_winding_jump":count});total+=n
    atomic_json(a.output,{"passed":True,"source_index_sha256":sha256_file(a.source/"index.json"),"frames":total,
        "horizon":a.horizon,"source_arrays_modified":False,"episodes":len(rows),
        "frames_with_artificial_winding_jump":sum(r["frames_with_artificial_winding_jump"] for r in report),
        "periodic_sincos_max_abs_error":periodic_error,"examples":examples,"rows":report,
        "new_model_SR":None,"FULL_FAST_codec_and_native_physics_still_required":True})
    print(json.dumps({"frames":total,"jump_frames":sum(r["frames_with_artificial_winding_jump"] for r in report),"examples":examples}))


if __name__ == "__main__":main()
