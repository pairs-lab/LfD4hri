"""Same120/model cohort with EXPLICIT saved-state recovery lineage for RDT."""
import argparse
from pathlib import Path

from VLA import run_cross_vla120 as queue
from VLA.report_cross_vla120 import render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdt-checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    original=queue.model_specs
    def specs():
        rows=original()
        for row in rows:
            if row["model"]=="rdt":
                row["checkpoint"]=str(args.rdt_checkpoint.resolve())
                row["continued_from_saved_R16_optimizer_updates"]=1000
                row["DataLoader_shuffle_restarted_NOT_bit_exact"]=True
        return rows
    queue.model_specs=specs
    plan=queue.initialize(args.output.resolve())
    plan["checkpoint_selection_rule"]="Saved R16 continuation to5000 for Smol/RDT, official bases otherwise; selection NOT based on evaluation outcomes"
    queue.atomic_json(args.output/"plan.json",plan)
    queue.run_queue(args.output,plan,48)
    import json
    (args.output/"REPORT.md").write_text(render(json.loads((args.output/"progress.json").read_text())))


if __name__ == "__main__":
    main()
