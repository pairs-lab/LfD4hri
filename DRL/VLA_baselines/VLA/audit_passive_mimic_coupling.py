"""Reuse fixed-trace diagnostic harness with explicitly bound passive gears."""
import argparse
import json
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main():
    import VLA.audit_mimic_lag_guard as harness
    from VLA.common.passive_mimic_coupling import install
    p=argparse.ArgumentParser(add_help=False);p.add_argument("--output",type=Path,required=True)
    args,_=p.parse_known_args()
    harness.install=install;harness.main()
    original=json.loads((args.output/"audit.json").read_text())
    atomic_json(args.output/"actual_passive_coupling_binding.json",{
        "scope":"passive_gear_FIXED_ACTION_mechanical_diagnostic_NOT_model_SR",
        "harness_report_sha256":sha256_file(args.output/"audit.json"),
        "harness_original_motor_guard_sha_is_not_the_used_experiment":True,
        "actual_implementation_sha256":sha256_file(Path(__file__).parent/"common/passive_mimic_coupling.py"),
        "harness_sha256":sha256_file(Path(harness.__file__)),"rows":original["rows"],
        "new_neural_controls":0,"production_qualified":False})


if __name__ == "__main__":main()
