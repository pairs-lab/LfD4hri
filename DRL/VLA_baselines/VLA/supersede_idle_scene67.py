"""Retire only an exact childless pre-training waiter after an input bug.

This does not signal a trainer, inference server, collector or episode.
The previous status/proofs remain unchanged; the separate handoff receipt
explains why a corrected immutable continuation must be launched.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve();statepath=output/'status.json'
    state=json.loads(statepath.read_text());pid=state['pid'];proc=Path('/proc')/str(pid)
    if state['stage']!='waiting_complete_physical_extraction' or state.get('history'):
        raise ValueError('Do not retire any started stage')
    tokens=(proc/'cmdline').read_bytes().split(b'\0');children=(proc/'task'/str(pid)/'children').read_text().strip()
    if (b'VLA.run_scene67_finetune_cycle' not in tokens or str(output).encode() not in tokens or children
        or (proc/'cwd').resolve()!=Path(__file__).resolve().parents[1]):
        raise ValueError('Not the exact childless pre-training waiter')
    receipt=output/'idle_input_bug_handoff.json'
    if receipt.exists():raise FileExistsError('Preserve previous handoff')
    digest=sha256_file(statepath);os.kill(pid,signal.SIGTERM);deadline=time.monotonic()+15
    while proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
        if time.monotonic()>deadline:raise TimeoutError('Idle waiter did not exit')
        time.sleep(.1)
    atomic_json(receipt,{'pid':pid,'children':children,'stage':state['stage'],'status_sha256':digest,
        'reason':'real corrective RGB fetch omitted the native current-observation singleton time axis',
        'no_training_or_neural_controls_started':True,'training_collection_or_evaluation_signalled':False,
        'original_failed_input_audit_preserved':True,'corrected_continuation_still_needs_real_batch_gate':True})
    print(json.dumps({'retired_idle_waiter':pid,'receipt':str(receipt)}))


if __name__=='__main__':main()
