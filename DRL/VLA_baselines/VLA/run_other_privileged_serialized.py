"""Preserve the existing native queue, reserve RDT's train->reload->eval gap.

Only its precisely verified childless WAITING coordinator is handed off.
No trainer, model server or in-flight robot episode is signalled. Inputs and
completed native controls are unchanged; this is a scheduling repair ONLY.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT
import VLA.run_other_privileged_vlas as original

OLD = ROOT / "VLA/audits/other_vla_privileged_r29_20261014/gpu_queue_cached_models_continuation_v2"
RDT = ROOT / "VLA/audits/rdt_privileged_postfit_r25_20261014"


def priority_pending(state, alive):
    if not state: return True
    # A saved checkpoint between stages does not release the reservation.
    return bool(alive or state.get("status") not in {"completed", "failed"})


def run(args):
    digest = sha256_file(Path(__file__))
    stock = original.available_gpu
    def available(model, **kwargs):
        if sha256_file(Path(__file__)) != digest: raise ValueError("Scheduling wrapper changed")
        state = json.loads((RDT / "status.json").read_text())
        proc = Path(f"/proc/{int(state['pid'])}")
        owner_alive = proc.exists() and (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        ready, gpu, allowed = stock(model, **kwargs)
        atomic_json(args.output.parent / "serialized_rdt_reservation.json", {"rdt_stage": state.get("stage"),
            "rdt_status": state.get("status"), "rdt_owner_alive": owner_alive,
            "reservation_pending": priority_pending(state, owner_alive), "wrapper_sha256": digest})
        return ready and not priority_pending(state, owner_alive), gpu, allowed
    original.available_gpu = available
    original.run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--launch", action="store_true")
    p.add_argument("--openvla-dataset", type=Path, required=True); p.add_argument("--openvla-parity", type=Path, required=True)
    p.add_argument("--native-prefix", type=Path, required=True); p.add_argument("--wait-hours", type=float, default=72)
    args = p.parse_args()
    if not args.launch: return run(args)
    args.output = args.output.resolve()
    if args.output.exists(): raise FileExistsError("New immutable continuation required")
    state = json.loads((OLD / "status.json").read_text()); pid = int(state["pid"]); proc = Path(f"/proc/{pid}")
    tokens = (proc / "cmdline").read_bytes().split(b"\0")
    children = (proc / "task" / str(pid) / "children").read_text().strip()
    if (state.get("stage") != "waiting_gpu_for_pi0" or state.get("status") != "waiting_gpu_no_training_or_episode_interruption"
        or b"VLA.run_other_privileged_vlas" not in tokens or str(OLD).encode() not in tokens
        or (proc / "cwd").resolve() != ROOT or children): raise ValueError("Not the exact childless waiting owner")
    for path, digest in state["frozen_files"].items():
        if sha256_file(path) != digest: raise ValueError("Frozen native inputs changed")
    atomic_json(OLD / "serialized_idle_handoff_receipt.json", {"old_pid": pid, "children": children,
        "old_status_sha256": sha256_file(OLD / "status.json"), "rdt_training_or_robot_episodes_interrupted": False,
        "reason": "Reserve RDT checkpoint/reload/evaluation before competing native GPU loads",
        "native_inputs_weights_completed_controls_unchanged": True, "new_output": str(args.output)})
    os.kill(pid, signal.SIGTERM)
    until = time.monotonic() + 15
    while proc.exists() and (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
        if time.monotonic() > until: raise TimeoutError("Waiting coordinator did not stop; no child signalled")
        time.sleep(.1)
    command = [sys.executable, "-m", "VLA.run_other_privileged_serialized", "--output", str(args.output),
        "--openvla-dataset", str(args.openvla_dataset.resolve()), "--openvla-parity", str(args.openvla_parity.resolve()),
        "--native-prefix", str(args.native_prefix.resolve()), "--wait-hours", str(args.wait_hours)]
    with args.output.with_name(args.output.name + "_launcher.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(args.output.with_name(args.output.name + "_launch.json"), {"pid": child.pid,
        "scheduling_only": True, "no_neural_process_signalled": True, "old_waiting_pid": pid})
    print(json.dumps({"pid": child.pid, "output": str(args.output)}))


if __name__ == "__main__": main()
