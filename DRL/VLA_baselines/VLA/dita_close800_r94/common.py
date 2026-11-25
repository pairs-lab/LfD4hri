from pathlib import Path
import signal

from VLA.common.provenance import sha256_file
from VLA.recovery_r85.common import ROOT, read, verify, utc
from VLA.dita4096_benchmark_r92.common import same, identity, alive
from VLA.benchmark_guardian_r89.common import live_children, parent_pid

OUT = ROOT/'VLA/audits/dita_close800_r94_20261023'
R86 = ROOT/'VLA/audits/continuous_r86_20261021'
R90 = ROOT/'VLA/audits/recovery_quality_r90_20261022'
CP = R86/'dita/fit10000'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
CPU = Path('/home/uet/miniconda3/envs/robot/bin/python')
PORT = 5994
SEEDS = list(range(20284001, 20284201))
TASKS = ('touch', 'pick', 'move', 'pick_place')


def bindings():
    files = list(Path(__file__).parent.glob('*.py'))
    for name in ('dita800_r87','receipt_recovery_r88','benchmark_guardian_r89',
                 'dita4096_benchmark_r92','recovery_quality_r90'):
        files += list((ROOT/'VLA'/name).glob('*.py'))
    return {str(p.resolve()): sha256_file(p) for p in files}


def module_name(expected):
    command = expected['command'] if expected else []
    return command[command.index('-m')+1] if '-m' in command else None


def proc_state(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def authorize_signal(expected, sig, role):
    """No optimizer may ever be signaled through this controller."""
    name = module_name(expected)
    if role == 'queue_parent':
        allowed = name == 'VLA.continuous_r86.owner' and sig in (signal.SIGSTOP,signal.SIGCONT,signal.SIGTERM)
    elif role == 'idle_recovery':
        allowed = name == 'VLA.recovery_quality_r90.owner' and sig == signal.SIGTERM
    elif role == 'own_benchmark':
        allowed = name == 'VLA.dita_close800_r94.evaluate' or (
            name == 'VLA.dita_transfer_r84.server' and str(OUT/'FROZEN_METHOD.json') in expected['command']
            and str(PORT) in expected['command'])
        allowed = allowed and sig in (signal.SIGTERM, signal.SIGKILL)
    elif role == 'own_reload':
        command = expected['command'] if expected else []
        allowed = (name == 'VLA.recovery_r85.dita_train' and 'reload' in command
            and 'fit' not in command and str(CP) in command
            and sig in (signal.SIGTERM, signal.SIGKILL))
    else:
        allowed = False
    if not allowed:
        raise ValueError('Signal is outside the explicitly owned scheduling/benchmark scope')


def reload_only(command):
    """The terminal Dita controller has no authorized fit command."""
    if command != ['VLA.recovery_r85.dita_train','reload','--checkpoint',str(CP)]:
        raise ValueError('Only a native reload of the exact final checkpoint is permitted')


def validate_checkpoint_metadata(meta):
    if meta.get('family') != 'dita' or not meta.get('completed') or meta.get('updates') != 10000:
        raise ValueError('The existing native Dita optimizer must commit all10,000 updates')
