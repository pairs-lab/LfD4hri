"""Finite repair-proof -> BOTH replay -> complete candidate, never a fit gate bypass."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "visibility-report", "repair", "output", "audit-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.output.exists() or args.audit_output.exists() or not 1 <= args.wait_seconds <= 7200:
        raise ValueError("Fresh candidate/audit outputs and finite wait required")
    args.audit_output.mkdir(parents=True)
    status_path = args.audit_output/"status.json"
    required = args.repair/"results.json"
    frozen = {str(path.resolve()): sha256_file(path) for path in
        (Path(__file__), Path(__file__).with_name("replace_transport_demonstrations.py"),
         Path(__file__).parent/"pi0/audit_closed_loop_replay.py", args.visibility_report,
         args.source/"index.json", args.source/"dataset_manifest.json")}
    def save(state, error=None):
        status_path.write_text(json.dumps({"status": state, "source": str(args.source.resolve()),
            "repair": str(args.repair.resolve()), "candidate": str(args.output.resolve()),
            "frozen_files": frozen, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_authorized": False, "error": error}, indent=2)+"\n")
    def verify():
        if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
            raise ValueError("Repair finalization source/implementation changed")
    try:
        save("waiting_for_actual_repair_results")
        deadline = time.monotonic()+args.wait_seconds
        while not required.is_file():
            verify()
            if time.monotonic() >= deadline:
                raise TimeoutError("Repair evidence wait expired")
            time.sleep(10)
        verify()
        evidence = json.loads(required.read_text())
        if not evidence.get("passed"):
            raise ValueError("Repair still fails actual execution or visibility; no silent promotion")
        replay = args.audit_output/"both_strict_replay.json"
        save("running_both_strict_physics_replay")
        subprocess.run([sys.executable, "-m", "VLA.pi0.audit_closed_loop_replay", "--source", str(args.repair),
            "--output", str(replay), "--mode", "both", "--native-success-mode", "strict", "--workers", "2"], check=True)
        verify()
        save("materializing_all_scenes_without_filtering")
        subprocess.run([sys.executable, "-m", "VLA.replace_transport_demonstrations", "--source", str(args.source),
            "--visibility-report", str(args.visibility_report), "--repair", str(args.repair),
            "--repair-replay", str(replay), "--output", str(args.output)], check=True)
        save("completed_candidate_NOT_qualified")
    except Exception as exc:
        save("stopped_no_silent_retry", str(exc))
        # Notify the waiting fit workflow even if no candidate was created.
        # This is a failure marker, NEVER a synthetic completion certificate.
        if not args.output.exists():
            args.output.mkdir(parents=True)
            (args.output/"collection_state.json").write_text(json.dumps({"status": "stopped_transport_repair",
                "error": str(exc), "training_authorized": False}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
