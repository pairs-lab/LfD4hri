import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VLA/audits/xvla800_r59_20261017'
CP = ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/xvla_round1/fit1500/checkpoints/step-014500'
SCHEDULE = ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/xvla_new_r1_global_schedule.json'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
MODEL_PY = ROOT/'VLA/.venvs/xvla/bin/python'
TASKS = ('touch', 'pick', 'move', 'pick_place')
SEEDS = list(range(20271001, 20271201))
PORT = 5907


def read(path): return json.loads(Path(path).read_text())


def resources():
    import subprocess
    mem = {r.split(':')[0]: int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    total, free = (int(v.strip()) for v in text.strip().split(','))
    return {'available_RAM_MiB': mem['MemAvailable']/1024, 'free_GPU_MiB': free, 'total_GPU_MiB': total}
