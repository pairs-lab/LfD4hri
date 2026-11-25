"""Report every admitted trial separately by stage and action, never fabricate SR."""
from pathlib import Path
from .common import OUT,read


def main():
    lines=['# Evo-1 native privileged adaptation — measured results','',
        'Same UR5e/Robotiq R62 plant, semantic first-hit metric and safety. ',
        'PUT is goal-region first-hit, not mandatory release/settle.',
        'Source: official LIBERO action checkpoint; stage2 uses disclosed VLM LoRA8.',
        'No Oracle action fallback or geometric residual controller.','']
    found=False
    for folder in sorted(OUT.iterdir()):
        path=folder/'progress.json'
        if not folder.is_dir() or not path.is_file():continue
        d=read(path)
        if not d.get('robot_SR_measured'):continue
        found=True;lines+=['## '+folder.name,'',f"Status: {d['status']}; {d['completed_trials']}/{d['planned_trials']} actual trials.",'',
            '|Action|Successes/trials|SR|','|---|---:|---:|']
        for task,v in d['per_action'].items():
            sr='not measured' if v['SR_percent'] is None else f"{v['SR_percent']:.1f}%"
            lines.append(f"|{task}|{v['successes']}/{v['n']}|{sr}|")
        lines+=['',f"Safety aborts: {sum(bool(r.get('safety_violation')) for r in d['rows'])}; retained in the denominator.",'']
    if not found:lines+=['No closed-loop success rate has been measured yet.','']
    lines+=['Development/prefix cohorts are selection data, not final800.',
        'One final800 cohort does not constitute five evaluated training seeds.','']
    target=OUT/'MEASURED_REPORT.md';pending=target.with_suffix('.md.pending')
    pending.write_text('\n'.join(lines));pending.replace(target)


if __name__=='__main__':main()
