"""Small durable index of measured results and queued experiments; no data copies."""
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.autonomy_r83.common import ROOT, OUT, R80, R82, TASKS, read, utc

REFERENCES = {
    'FLOWER': 'VLA/audits/flower800_r77_20261019',
    'X-VLA': 'VLA/audits/xvla800_r59_20261017',
    'SmolVLA': 'VLA/audits/cross_vla800_r62_20261018/smolvla',
    'pi0': 'VLA/audits/cross_vla800_r62_20261018/pi0',
    'VLA-Adapter': 'VLA/audits/cross_vla800_r62_20261018/vla_adapter',
    'VLA-JEPA': 'VLA/audits/cross_vla800_r62_20261018/statistical_review_r64',
    'Evo-1_current800': str(R82.relative_to(ROOT)),
    'Evo-1_training_and_DEV': str(R80.relative_to(ROOT)),
    'InternVLA': 'VLA/audits/internvla_native_r72_20261019',
    'method_training_protocol_table': 'VLA/audits/method_protocol_table_r79_20261020/REPORT.md',
    'qualitative_main_videos': 'VLA/qualitative/ur5e_xvla_put_r60_named_failures_20261017/index.html',
}


def refresh(state):
    atomic_json(OUT / 'CATALOG.json', {'updated_UTC': utc(), 'references': {
        name: {'path': str(ROOT / path), 'exists': (ROOT / path).exists(),
               'note': 'Pointer only: not a claim of completed800 for every method'}
        for name, path in REFERENCES.items()}, 'automation': state})
    lines = ['# Autonomous VLA audit → adapt → evaluate', '',
             f"Updated UTC: {utc()}. Status: {state['status']}; stage: {state['stage']}.", '',
             'Current Evo-1 R82 final800 continues unchanged; this owner never edits or stops it.',
             'Frozen FLOWER/X-VLA and all previous measurements remain preserved.', '',
             '## Active / queued jobs', '', '| Model | State | Details |', '|---|---|---|']
    for name, value in state.get('jobs', {}).items():
        lines.append(f"| {name} | {value.get('status')} | {value.get('stage', value.get('reason', ''))} |")
    progress = R82 / 'episodes/progress.json'
    if progress.exists():
        p = read(progress)
        lines += ['', f"Existing Evo-1 R82 actual progress: {p['completed_trials']}/800. See its own report for outcomes."]
    for name in ('smolvla', 'vla_adapter', 'evo1'):
        folder = OUT / name
        if not folder.exists():
            continue
        stages = sorted(folder.glob('*/progress.json'))
        if not stages:
            continue
        lines += ['', '## ' + name, '', '| Stage | Actual completed | Per-action successes/n (touch, pick, move, put) |',
                  '|---|---:|---|']
        for path in stages:
            p = read(path)
            per = p.get('per_action', {})
            text = '; '.join(f"{per.get(t, {}).get('successes', 0)}/{per.get(t, {}).get('n', 0)}" for t in TASKS)
            lines.append(f"| {path.parent.name} | {p['completed_trials']}/{p['planned_trials']} | {text} |")
    lines += ['', '## Rules and limits', '',
              '- CPU audit and actual optimizer2/fresh reload must pass before long adaptation.',
              '- SmolVLA and VLA-Adapter use task/target/rotation/grasp-balanced TRAIN and explicit real-time padding.',
              '- Maximum new updates per adapted model1536 (checkpoints2/512/1536); no unbounded refit loop.',
              '- A revised candidate must beat its unchanged baseline on DEV and reach20/40, at least2/task.',
              '- Independent CONF40 must also reach20/40 and2/task before its fresh secondary800.',
              '- No tuning on final800, no relaxed success, no dropped safety failures or fake multi-seed values.',
              '- R83 uses a fresh secondary800 cohort, not an untouched repeat of previously seen R62 final scenes.',
              '- Integration errors are quarantined, not counted as model SR0; the next implemented model continues.',
              '- No automatic deletion, package upgrade or arbitrary generated code patch in the background.',
              '- A detached local owner survives this chat turn, not a machine shutdown. Terminal states remain recorded.', '',
              '## Where to find artifacts', '',
              '`owner_status.json`: current owner/worker PIDs, jobs and heartbeat. `EVENTS.jsonl`: phase journal.',
              '`CATALOG.json`: links to all selected historical models, protocols and qualitative gallery.',
              '`CPU_AUDIT.json`, `GEOMETRY_EXCLUSION.json`, `IMAGE_REVIEW.json`: preflight evidence.',
              '`<model>/fit*/`: learned weights, exact optimizer/RNG, input stats, update logs and reload proof.',
              '`<model>/<stage>/`: every episode outcome, actual neural queries and full state/action traces.',
              '`<model>/SELECTION.json`: DEV-only decision; `<model>/FROZEN_FINAL.json`: final checkpoint/method hashes.',
              '`<model>/FINAL_AUDIT.json`: coverage and action-source verification after800; no final media.', '',
              '## Existing results / protocol / qualitative index', '']
    for name, path in REFERENCES.items():
        lines.append(f'- [{name}]({ROOT / path})')
    pending = OUT / 'REPORT.md.tmp'
    pending.write_text('\n'.join(lines) + '\n')
    pending.replace(OUT / 'REPORT.md')
