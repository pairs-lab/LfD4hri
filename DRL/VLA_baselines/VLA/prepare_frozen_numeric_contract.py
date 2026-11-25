"""TRAIN-only exact numeric/sampling preparation; no data qualification/GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.lerobot.recompute_exact_stats import exact
from VLA.smolvla.balanced_sampling import frame_weights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--train-audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train=args.train.resolve()
    manifest=json.loads((train/"dataset_manifest.json").read_text())
    if manifest.get("split")!="train":
        raise ValueError("Only frozen TRAIN may determine normalization")
    audit=json.loads(args.train_audit.read_text())
    if (not audit["passed"] or audit["failures"] or audit["dataset"]!=str(train)
            or audit["index_sha256"]!=sha256_file(train/"index.json")
            or audit["manifest_sha256"]!=sha256_file(train/"dataset_manifest.json")):
        raise ValueError("Changed or failed TRAIN audit")
    index=json.loads((train/"index.json").read_text())
    validation=json.loads((args.validation/"index.json").read_text())
    if {r["scene_group"] for r in index}&{r["scene_group"] for r in validation}:
        raise ValueError("Physical scene overlap")
    states,actions=[],[]
    for row in index:
        with np.load(train/row["file"],allow_pickle=False) as arrays:
            state=np.c_[arrays["joint_positions"],arrays["gripper_closure"]].astype(np.float64)
            action=arrays["canonical_actions"].astype(np.float64)
            if state.shape!=action.shape or state.shape!=(row["steps"],7):
                raise ValueError("Unsynchronized TRAIN state/action shapes")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Nonfinite TRAIN values")
            states.append(state); actions.append(action)
    state=np.concatenate(states); action=np.concatenate(actions)
    weights=frame_weights(index,.30,True)
    draws=5000*16
    expected_unique=float(np.sum(-np.expm1(draws*np.log1p(-weights))))
    report={"scope":"CPU TRAIN-only exact numeric/sampling PREPARATION, not converted-dataset or training qualification",
        "ready_to_train":False,"training_authorized":False,"gpu_or_optimizer_executed":False,
        "train_source":str(train),"train_index_sha256":sha256_file(train/"index.json"),
        "train_manifest_sha256":sha256_file(train/"dataset_manifest.json"),
        "train_audit_sha256":sha256_file(args.train_audit),
        "validation_index_sha256":sha256_file(args.validation/"index.json"),
        "validation_used_for_normalization":False,"scene_overlap":0,
        "episodes":len(index),"frames":len(state),
        "state_semantics":"observed six UR5e joint angles plus encoder/FK-calibrated Robotiq aperture closure",
        "action_semantics":"executed world delta xyz metres, tool delta roll/pitch/yaw radians, requested gripper -1 open/+1 close",
        "exact_numeric_stats":{"observation.state":exact(state),"action":exact(action)},
        "constant_action_dimensions":np.flatnonzero(np.ptp(action,axis=0)==0).tolist(),
        "sampling_proposal":{"optimizer_updates":5000,"accumulation":16,"batch_size":1,"action_horizon":5,
            "anchor_draws":draws,"replacement_sampling":True,"initial_probability":.30,
            "initial_anchor_expected_draws":.30*draws,"expected_unique_anchors":expected_unique,
            "note":"Sampling budget is NOT a data-sufficiency guarantee; recovery/compositional gaps remain"},
        "remaining_gates":["Broad corrected touch/recovery quality and unchanged frozen split",
            "Immutable qualified TRAIN source and model-specific image review",
            "Fresh final-source PNG/Parquet/actual-loader roundtrip and exact converted TRAIN statistics",
            "Deployment sensor/action parity, CUDA memory and two-update FP32-master smoke",
            "Fresh-base bounded fit then held-out four-task rollouts/video"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("episodes","frames","scene_overlap","ready_to_train","sampling_proposal")}),flush=True)


if __name__=="__main__":
    main()
