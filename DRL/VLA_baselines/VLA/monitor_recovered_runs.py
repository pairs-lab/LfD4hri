"""Detached low-memory liveness/receipt monitor; no model imports or control."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
R17=ROOT/"VLA/audits/recovery_r17_20261013"
R18=ROOT/"VLA/audits/recovery_r18_horizon_fix_20261014"
DEST=ROOT/"VLA/audits/active_repair_cycles_20261013.json"
CROSS=ROOT/"VLA/audits/cross_vla120_per_model_20261013"


def atomic(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}


def owner_alive(pid,module):
    if type(pid) is not int or pid<1: return False
    proc=Path(f"/proc/{pid}")
    try:
        tokens=(proc/"cmdline").read_bytes().split(b"\0")
        module_match=any(tokens[i:i+2]==[b"-m",module.encode()] for i in range(len(tokens)-1))
        return module_match and (proc/"stat").read_text().rsplit(")",1)[1].split()[0]!="Z"
    except (FileNotFoundError,IndexError): return False


def tail(path,size=32768):
    if not path.exists(): return ""
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-size));return stream.read().decode(errors="replace")


def snapshot():
    first,current=read(R17/"status.json"),read(R18/"status.json")
    smol=first.get("lanes",{}).get("smol_resume",{})
    rdt=current.get("lanes",{}).get("rdt_continue",{})
    numbers=re.findall(r"(\d+)/64000",tail(R17/"smol_resume.log"))
    smol_new=int(numbers[-1])//16 if numbers else 0
    rdt_steps=0
    # A live trainer keeps its original log after coordinator ownership moves.
    # Prefer its actual metrics file, not a nonexistent new-driver stdout path.
    metrics=ROOT/"VLA/rdt/body_v9_recovery_r17_20261013/test1/train/train_metrics.jsonl"
    for line in reversed(tail(metrics).splitlines()):
        try: item=json.loads(line)
        except json.JSONDecodeError: continue
        if "step" in item: rdt_steps=item["step"];break
    projections=read(R17/"train/progress.json")
    trials=read(R18/"development120/progress.json")
    result={"scope":"Actual process liveness and saved-source recovery; no success inferred from losses/replays",
        "monitor_pid":os.getpid(),"updated_at_utc":datetime.now(timezone.utc).isoformat(),
        "recovery_cycle":str(R18/"status.json"),"recovery_owner_alive":owner_alive(current.get("pid"),"VLA.recover_interrupted_cycles"),
        "recovery_stage":current.get("stage"),"recovery_error":current.get("error"),
        "models":{},"causal_projection":{"completed_episodes":projections.get("completed_episodes"),
            "planned_episodes":2107,"status":projections.get("status"),"TRAIN_frames":projections.get("frames")},
        "last_completed_SmolVLA_R14_successes":1,"last_completed_SmolVLA_R14_trials":40,
        "old_dead_owner_statuses_NOT_live":True,"learned_simstate_trials":trials.get("completed_trials",0),
        "simstate_SR120":trials.get("successes",0)/120 if trials.get("status")=="completed120_privileged_input_VLA" else None,
        "privileged_inputs_reported_separately":True,"final_test_used":False,"stable_high_success_established":False}
    for name,data,new_count,module in (("smolvla",smol,smol_new,"VLA.smolvla.train_phase_balanced"),
                                      ("rdt",rdt,rdt_steps,"VLA.rdt.continue_interrupted_training")):
        alive=owner_alive(data.get("pid"),module)
        result["models"][name]={"pid":data.get("pid"),"owner_alive":alive,
            "status":"optimizing_saved_checkpoint_continuation" if alive else "not_optimizing_check_saved_artifacts",
            "saved_start_updates":1000,"logged_new_updates":new_count,"logged_total_estimate":1000+new_count,
            "intended_total_updates":5000,"logged_total_NOT_new_saved_Adam_counter":True,
            "completed_new_development_success_rate":None}
    return result


def report(state):
    local=datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
    lines=["# Cross-VLA: 120 episodes/model, 30/action", "",
        f"Live check: {local:%Y-%m-%d %H:%M:%S %Z}. Pending is NOT 0%.","",
        "Old queue owner is extinct: old running JSON is NOT live. Recovery has a detached owner, periodic checkpoints and independent failure records.",
        f"Current recovery: `{R18.relative_to(ROOT)}/status.json`.","",
        "| Fitting model | Owner alive | Saved start | Logged total estimate | Target | New learned SR |",
        "|---|---|---|---|---|---|"]
    for name,row in state["models"].items():
        lines.append(f"| {name} | {row['owner_alive']} | 1000 | {row['logged_total_estimate']} (NOT saved-counter proof) | 5000 | pending |")
    data=state["causal_projection"]
    lines += ["",f"Causal TRAIN projection: {data['completed_episodes']}/2107; status `{data['status']}`.",
        "The new extractor passed8 real cases/407 frames with state18 arrays EXACT to the original pilot. Full TRAIN/477 validation and actual-loader gates remain mandatory.","",
        "Completed historical SmolVLA R14:1/40 (2.5%); not a new result, not model-weakness proof.",
        "Standard six-model120 budget remains720; pi0/FAST/OpenVLA/CogACT are still official ZERO-SHOT diagnostics, not matched post-fit paper comparisons.",
        "SmolVLA+SimState is separate: privileged requested-target/goal/grasp inputs, neural controls only; no oracle fallback or scorer relaxation.",
        "Original data/results and all observed failures are preserved. Safety aborts count as task failures; integration errors remain incomplete.", ""]
    cross=ROOT/("VLA/audits/cross_vla120_"+R18.name)
    if (cross/"REPORT.md").exists():
        lines += [f"Actual newly executed model cohort: `{cross.relative_to(ROOT)}/REPORT.md`.",""]
    return "\n".join(lines)


def main():
    global R18
    if "--cycle" in sys.argv:
        R18=Path(sys.argv[sys.argv.index("--cycle")+1]).resolve()
    if "--launch" in sys.argv:
        marker=R18/"evidence_monitor_launch.json"
        previous=read(marker)
        if previous and ("--replace-dead-monitor" not in sys.argv or owner_alive(previous.get("pid"),"VLA.monitor_recovered_runs")):
            raise FileExistsError("Do not start duplicate live monitors")
        history=previous.get("history",[])+([previous["pid"]] if previous else [])
        log_path=R18/("evidence_monitor.log" if not previous else f"evidence_monitor_replacement{len(history)}.log")
        with log_path.open("x") as log:
            process=subprocess.Popen([sys.executable,"-m","VLA.monitor_recovered_runs","--cycle",str(R18)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic(marker,{"pid":process.pid,"detached_session":True,"history":history});print(process.pid);return
    deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline:
        state=snapshot();atomic(DEST,state)
        temp=CROSS/"REPORT.md.tmp";temp.write_text(report(state));temp.replace(CROSS/"REPORT.md")
        if not state["recovery_owner_alive"]:
            state["monitor_status"]="recovery_owner_not_live_review_required";atomic(DEST,state);return
        time.sleep(30)


if __name__=="__main__":
    main()
