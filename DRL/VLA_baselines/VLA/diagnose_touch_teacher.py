"""Reproduce one frozen TRAIN touch case with boundary contact/motion traces."""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchCollection,SurfaceLimitedTouchTeacher


def contacts(env):
    closest=p.getClosestPoints(env.gripper.body,env.pick_id,distance=.025)
    actual=p.getContactPoints(env.gripper.body,env.pick_id)
    return {"gap_m":min((c[8] for c in closest),default=None),
        "forces_n":[c[9] for c in actual],
        "closest":[{"link":c[3],"gap":c[8],"normal":list(c[7])} for c in closest]}


class TracedCollection(SurfaceLimitedTouchCollection):
    def __init__(self,*args):
        super().__init__(*args)
        self.trace=[]

    def step(self,action):
        before=contacts(self.env)
        result=super().step(action)
        self.trace.append({"frame":len(self.trace),"action":np.asarray(action).tolist(),
            "before":before,"after":contacts(self.env),"strict_success":result[3]["strict_raw_success"],
            "quality_success":result[3]["success"],"safety_violation":result[3].get("safety_violation")})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--train-episodes",type=Path,required=True)
    parser.add_argument("--episode",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    allowed={r["file"]:r for r in json.loads(args.train_episodes.read_text()) if Path(r["source"]).resolve()==source}
    if args.episode not in allowed or sha256_file(source/args.episode)!=allowed[args.episode]["sha256"]:
        raise ValueError("No diagnostic labels from validation/test or changed TRAIN")
    row=next(r for r in json.loads((source/"index.json").read_text()) if r["file"]==args.episode)
    if row["task"]["name"]!="touch":
        raise ValueError("Touch only")
    # collect_episode also records camera audit masks; render them even when
    # this diagnostic only retains motion/contact traces.
    with audit_environment(source,image_size=32,
                           required_success_steps=1,use_semantic_success=False) as env:
        proxy=TracedCollection(env,source,row)
        _,meta=collect_episode(proxy,TaskSpec.from_dict(row["task"]),policy_factory=SurfaceLimitedTouchTeacher)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"scope":"One frozen TRAIN expert diagnostic; not training data",
        "source":str(source),"episode":args.episode,"payload_sha256":allowed[args.episode]["sha256"],
        "metadata":meta,"trace":proxy.trace},indent=2)+"\n")
    print(json.dumps({"success":meta["success"],"steps":meta["steps"],"last":proxy.trace[-1]}),flush=True)


if __name__=="__main__":
    main()
