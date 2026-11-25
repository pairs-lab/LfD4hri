"""Real eight-case equality gate for the throughput-fixed causal extractor."""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.resume_simstate_projection import extract_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-pilot",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve prior actual pilot")
    old=json.loads((args.previous_pilot/"audit.json").read_text())
    if not old["passed"] or not old["pilot_only"] or len(old["rows"])!=8:
        raise ValueError("Eight-case actual old pilot required")
    source=Path(old["source"])
    index={r["file"]:r for r in json.loads((source/"index.json").read_text())}
    args.output.mkdir(parents=True);(args.output/"states").mkdir()
    results=[]
    try:
        for original in old["rows"]:
            proof=extract_episode(source,args.output,index[original["file"]])
            with (np.load(args.previous_pilot/"states"/proof["file"],allow_pickle=False) as before,
                  np.load(args.output/"states"/proof["file"],allow_pickle=False) as after):
                if not np.array_equal(before["state"],after["state"]):
                    raise ValueError("Materialization/worker refactor changed causal feature values")
            results.append(proof)
            print(f"Real fast-extractor pilot {len(results)}/8: {proof['file']}",flush=True)
        atomic_json(args.output/"audit.json",{"passed":True,"pilot_only":True,"qualified_for_training":False,
            "episodes":8,"frames":sum(r["frames"] for r in results),"all_state18_values_byte_exact_to_original_pilot":True,
            "strict_physics_sensors_and_feedback_passed":True,"rows":results,
            "original_pilot_audit_sha256":sha256_file(args.previous_pilot/"audit.json"),
            "neural_success_established":False})
    except Exception as error:
        atomic_json(args.output/"audit.json",{"passed":False,"error":str(error),"rows":results,"qualified_for_training":False})
        raise


if __name__=="__main__":
    main()
