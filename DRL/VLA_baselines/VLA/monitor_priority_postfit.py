"""Read-only detached evidence monitor, including per-model live SR counts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def alive(pid, cycle=None):
    try:
        proc = Path(f"/proc/{pid}")
        tokens = (proc / "cmdline").read_bytes().split(b"\0")
        if any(module in tokens for module in (b'VLA.evaluate_smolvla_flow20_r36',b'VLA.run_target_crop_r36',b'VLA.run_flow20_extension_r36',b'VLA.run_flow20_parallel_r37',b'VLA.audit_native_policy_fit_r37',b'VLA.evaluate_contactstop_smol_r37',b'VLA.evaluate_bounded_feedback_r38',b'VLA.run_smol_priority_r38',b'VLA.audit_smol_fit_r38',b'VLA.evaluate_smol800_r39',b'VLA.resume_smol800_r40',b'VLA.probe_smol_train_now_r40',b'VLA.evaluate_smol_chunk2_r41',b'VLA.extend_smol_chunk2_r41')):
            return ((proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
                and (proc/'cwd').resolve()==ROOT and (cycle is None or str(cycle.resolve()).encode() in tokens))
        return ((proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
            and any(module in tokens for module in (b"VLA.run_priority_postfit_cycle", b"VLA.run_privileged_priority_cycle", b"VLA.run_rdt_simstate_cycle", b"VLA.run_other_privileged_vlas", b"VLA.run_other_privileged_serialized", b"VLA.run_scene64_repair_cycle", b"VLA.run_scene67_finetune_cycle", b"VLA.run_scene32_openpi_cycle", b"VLA.run_scene64_other_input_gates", b"VLA.run_native_recovery_view_cycle", b"VLA.run_all_privileged_r33", b"VLA.run_openpi_eef_repair_r33", b"VLA.run_scene67_resume_cycle", b"VLA.run_fast_codec_preflight", b"VLA.run_openvla_balanced_recovery_r34", b"VLA.run_scene67_parallel_r34"))
            and (cycle is None or str(cycle.resolve()).encode() in tokens))
    except (FileNotFoundError, IndexError): return False


def atomic(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text); temp.replace(path)


def process_state(pid):
    try:return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]
    except (FileNotFoundError,IndexError):return None


def secondary_rdt_snapshot():
    cycle = ROOT/"VLA/audits/rdt_privileged_postfit_r25_20261014"
    state = read(cycle/"status.json")
    if not state:
        return {}
    metrics = ROOT/"VLA/rdt_simstate"/cycle.name/"train/train_metrics.jsonl"
    latest = {}
    if metrics.exists():
        with metrics.open("rb") as stream:
            stream.seek(max(0, metrics.stat().st_size-16384))
            for line in reversed(stream.read().decode(errors="replace").splitlines()):
                try:
                    latest = json.loads(line); break
                except json.JSONDecodeError:
                    pass
    neural = read(cycle/"development120/progress.json")
    return {"cycle":str(cycle),"owner_alive":alive(state.get("pid",0),cycle),"stage":state.get("stage"),
        "status":state.get("status"),"error":state.get("error"),"logged_updates_NOT_saved_proof":latest.get("step"),
        "saved_precision":read(cycle/"saved_precision.json"),"privileged_model_inputs":True,
        "completed_trials":neural.get("completed_trials",0),"successes":neural.get("successes",0)}


def monitor(cycle):
    plan = read(cycle / "plan.json")
    cross = Path(plan["cross_root"])
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        state = read(cycle / "status.json")
        progress = read(cross / "progress.json")
        neural = read(cycle / "development120/progress.json")
        owner = alive(state.get("pid", 0), cycle)
        child = read(cycle / "active_child.json")
        rdt = secondary_rdt_snapshot()
        other_root = ROOT/"VLA/audits/other_vla_privileged_r29_20261014"
        other_queue = other_root/"gpu_queue_rdt_serialized_20261015"
        other_state = read(other_queue/"status.json")
        other_neural = read(other_queue/"openvla_development120/progress.json")
        other = {"queue":str(other_queue), "owner_alive":alive(other_state.get("pid",0),other_queue),
            "stage":other_state.get("stage"),"status":other_state.get("status"),"error":other_state.get("error"),
            "models_native_integration_NOT_task_SR":other_state.get("models",{}),
            "openvla_privileged_pilot_completed_episodes":other_neural.get("completed_trials",0),
            "openvla_privileged_pilot_successes":other_neural.get("successes",0)}
        scene_root = ROOT/"VLA/audits/scene64_recovery_r30_20261014"
        scene_queue = scene_root/"automatic_data_cycle"
        scene_state = read(scene_queue/"status.json")
        full64 = read(scene_root/"full_train64/progress.json")
        recovery40 = read(scene_root/"recovery40/progress.json")
        recovery_replay = read(scene_root/"recovery40_replay/audit.json")
        scene = {"owner_alive":alive(scene_state.get("pid",0),scene_queue),
            "stage":scene_state.get("stage"),"status":scene_state.get("status"),"error":scene_state.get("error"),
            "full_original_source64_replayed_episodes":full64.get("completed_episodes",0),
            "full_original_source64_frames":full64.get("frames",0),
            "actual_student_TRAIN_prefix_attempts":recovery40.get("completed_attempts",0),
            "expert_correction_successes_NOT_VLA_SR":recovery40.get("expert_correction_successes_NOT_student_SR",0),
            "correction_labeled_frames":recovery40.get("labeled_frames",0),
            "correction_replay_passed":recovery_replay.get("passed"),"training_qualified":False,
            "new64_model_finetuned":False,"new64_model_SR":None}
        fit_root = ROOT/"VLA/audits/scene67_deadline_r31_20261015/automatic_full_gate_fit_v2"
        fit_state = read(fit_root/"status.json")
        fit_progress = read(fit_root/"development120/progress.json")
        fit = {"cycle":str(fit_root),"owner_alive":alive(fit_state.get("pid",0),fit_root),
            "stage":fit_state.get("stage"),"status":fit_state.get("status"),"error":fit_state.get("error"),
            "history":fit_state.get("history",[]),"completed_trials":fit_progress.get("completed_trials",0),
            "successes":fit_progress.get("successes",0),
            "new_model_success_rate_established":fit_progress.get("status")=="completed120_privileged_input_VLA",
            "completed_development_SR120":fit_progress.get("successes",0)/120
                if fit_progress.get("status")=="completed120_privileged_input_VLA" else None}
        other_input_root = fit_root.parent/"other_native_full_input_gates"
        other_input_state = read(other_input_root/"status.json")
        other_inputs = {"cycle":str(other_input_root),"stage":other_input_state.get("stage"),
            "owner_alive":alive(other_input_state.get("pid",0),other_input_root),
            "status":other_input_state.get("status"),"error":other_input_state.get("error"),
            "history":other_input_state.get("history",[]),"GPU_allocations":0,"model_SR":None}
        native32_root=fit_root.parent/"openpi_native32_actual_fit_r32"
        native32_state=read(native32_root/"status.json")
        native32={"cycle":str(native32_root),"owner_alive":alive(native32_state.get("pid",0),native32_root),
            "process_state":process_state(native32_state.get("pid",0)),"stage":native32_state.get("stage"),
            "status":native32_state.get("status"),"error":native32_state.get("error"),
            "models":native32_state.get("models",{}),"high_level_actions_neural_only":True}
        native32['orphan_process_audit']=read(native32_root/'ORPHAN_PROCESS_AUDIT_r33.json')
        native32['FAST_old_smoke_invalidated_by_real_zero_fallback']=read(native32_root/'FAST_DECODER_AUDIT_INVALIDATES_OLD_SMOKE.json')
        if not native32['owner_alive'] and native32.get('status')=='running':
            native32['status']='ORPHANED_old_status_NOT_live_training_or_completion'
        repair_root=fit_root.parent/'all_family_privileged_r33_v4';repair_state=read(repair_root/'status.json')
        repair={'cycle':str(repair_root),'owner_alive':alive(repair_state.get('pid',0),repair_root),
            **{k:repair_state.get(k) for k in ('stage','status','error','models','history')},
            'native_recovery_admission_passed':read(repair_root/'native_view_admission.json').get('passed'),
            'OpenVLA_actual_two_update_precision':read(repair_root/'openvla_precision_2.json'),
            'development':{m:{k:v for k,v in read(repair_root/(m+'_development120/progress.json')).items()
                if k not in ('rows','controls')} for m in ('openvla','rdt','cogact')}}
        eef_root=fit_root.parent/'openpi_EEF_repair_r33_v4';eef_state=read(eef_root/'status.json')
        eef={'cycle':str(eef_root),'owner_alive':alive(eef_state.get('pid',0),eef_root),
            **{k:eef_state.get(k) for k in ('stage','status','error','models','history')},
            'development':{m:{k:v for k,v in read(eef_root/(m+'_EEF_development120/progress.json')).items()
                if k not in ('rows','controls')} for m in ('pi0','pi0_fast')}}
        resumed_root=fit_root.parent/'scene67_resume_parallel_r34'
        if not resumed_root.exists():resumed_root=fit_root.parent/'scene67_resume_r33_v2'
        resumed_state=read(resumed_root/'status.json')
        resumed_development=Path(resumed_state['active_attempt'])/'development120' if resumed_state.get('active_attempt') else resumed_root/'development120'
        resumed={'cycle':str(resumed_root),'owner_alive':alive(resumed_state.get('pid',0),resumed_root),
            **{k:resumed_state.get(k) for k in ('stage','status','error')},
            'development':{k:v for k,v in read(resumed_development/'progress.json').items() if k not in ('rows','controls')},
            'memory_journal':read(Path(resumed_state['active_attempt'])/'applied_step_resource_journal.json') if resumed_state.get('active_attempt') else {}}
        codec_root=fit_root.parent/'FAST32_EEF_codec_physical_r33';codec_state=read(codec_root/'status.json')
        codec={'cycle':str(codec_root),'owner_alive':alive(codec_state.get('pid',0),codec_root),
            **{k:codec_state.get(k) for k in ('stage','status','error','offline_physical_passed')},'model_SR_established':False}
        balanced_pointer=read(fit_root.parent/'OpenVLA_balanced_recovery_active.json')
        balanced_root=Path(balanced_pointer.get('cycle',str(fit_root.parent/'OpenVLA_balanced_recovery_r34_v2')))
        if balanced_root.parent!=fit_root.parent:raise ValueError('Unexpected motion-refinement status root')
        if not balanced_root.exists():balanced_root=fit_root.parent/'OpenVLA_balanced_recovery_r34'
        balanced_state=read(balanced_root/'status.json')
        balanced={'cycle':str(balanced_root),'owner_alive':alive(balanced_state.get('pid',0),balanced_root),
            **{k:balanced_state.get(k) for k in ('stage','status','error')},
            'other_VLA_families_have_GPU_priority':True,
            'native128_batch2_passed':read(balanced_root/'native128_batch2/audit.json').get('passed'),
            'native128_batch1_passed':read(balanced_root/'native128_batch1/audit.json').get('passed'),
            'development40':{k:v for k,v in read(balanced_root/'development40/progress.json').items() if k not in ('rows','controls')}}
        def r36_cycle(root):
            state=read(root/'status.json');development=read(root/'progress.json')
            if (root/'development120/progress.json').exists():development=read(root/'development120/progress.json')
            return {'cycle':str(root),'owner_alive':alive(state.get('pid',0),root),
                **{k:state.get(k) for k in ('pid','stage','status','error')},
                'development':{k:v for k,v in development.items() if k not in ('rows','controls')}}
        flow20=r36_cycle(fit_root.parent/'Smol_native20_r36')
        flow20['actual_baseline10_reproduced8']=read(fit_root.parent/'Smol_native20_r36/baseline10_actual8.json').get('passed')
        flow20['actual_native20_qualified8']=read(fit_root.parent/'Smol_native20_r36/native20_actual8.json').get('passed')
        crop_pointer=read(fit_root.parent/'Smol_GTcrop_motion_r36_active.json')
        crop_root=Path(crop_pointer.get('cycle',str(fit_root.parent/'Smol_GTcrop_motion_r36_v3')))
        if crop_root.parent!=fit_root.parent:raise ValueError('Unexpected GTcrop status pointer')
        crop=r36_cycle(crop_root)
        crop['native3view128_passed']=read(fit_root.parent/'target_crop_r36_native128_v2/audit.json').get('passed')
        crop['full_actual_mask_coverage']=read(fit_root.parent/'target_crop_r36_all_masks/audit.json').get('counts')
        crop['native_trainer_entry_preGPU']=read(crop_root/'native_trainer_entry_preGPU.json')
        extension_pointer=read(fit_root.parent/'Smol_native20_extension_r36_active.json')
        extension_root=Path(extension_pointer.get('cycle',str(fit_root.parent/'Smol_native20_extension_r36_v2')))
        if extension_root.parent!=fit_root.parent:raise ValueError('Unexpected native20 status pointer')
        flow_extension=r36_cycle(extension_root)
        flow_extension['decision']=read(extension_root/'decision.json')
        native_recovery_root=fit_root.parent/"native_corrective_full38_r32"
        native_recovery_state=read(native_recovery_root/"status.json")
        native_recovery={"cycle":str(native_recovery_root),
            "owner_alive":alive(native_recovery_state.get("pid",0),native_recovery_root),
            "stage":native_recovery_state.get("stage"),"status":native_recovery_state.get("status"),
            "error":native_recovery_state.get("error"),"models":native_recovery_state.get("models",{}),
            "collection":{k:v for k,v in read(native_recovery_root/"fresh38/progress.json").items() if k!="rows"},
            "data_view_admission_passed":read(native_recovery_root/"native_view_admission.json").get("passed"),
            "model_SR_established":False,"long_fit_qualified":False}
        render_wait_path=fit_root.parent/"user_render_priority_wait_r33/status.json"
        if not render_wait_path.exists():render_wait_path=fit_root.parent/"user_render_priority_wait_r32_generation2/status.json"
        if not render_wait_path.exists():render_wait_path=fit_root.parent/"user_render_priority_wait_r32/status.json"
        render_wait=read(render_wait_path)
        from VLA.wait_render_then_resume_queues import render_jobs
        live_render_pids=render_jobs()
        render_wait={**render_wait,'live_render_or_encoder_pids':live_render_pids,
            'live_render_count':len(live_render_pids)}
        log_path = cycle / "simstate_train.log"
        microsteps = []
        if log_path.exists():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 32768))
                microsteps = re.findall(r"(\d+)/40000", log.read().decode(errors="replace"))
        snapshot = {"monitor_pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "recovery_cycle": str(cycle / "status.json"), "recovery_owner_alive": owner,
            "recovery_stage": state.get("stage"), "recovery_error": state.get("error"),
            "standard_cohort": str(cross / "progress.json"),
            "completed_standard_trials": progress.get("completed_neural_trials", 0),
            "models": {m: {k: row.get(k) for k in ("status", "adaptation_regime", "completed_trials", "successes", "success_rate_for_completed120", "by_task", "error")}
                for m, row in progress.get("models", {}).items()},
            "simstate_completed_trials": neural.get("completed_trials", 0),
            "simstate_provisional_successes": neural.get("successes", 0),
            "simstate_provisional_SR_NOT_completed120": (neural.get("successes",0)/neural["completed_trials"]
                if neural.get("completed_trials",0) else None),
            "rdt_privileged_cycle_SEPARATE_from_standard_RDT":rdt,
            "privileged_active_child": child,
            "simstate_logged_optimizer_updates_NOT_saved_proof": int(microsteps[-1]) // 8 if microsteps else None,
            "simstate_SR120": neural.get("successes", 0) / 120 if neural.get("status") == "completed120_privileged_input_VLA" else None,
            "privileged_inputs_reported_separately": True, "stable_high_success_established": False}
        atomic(ROOT / "VLA/audits/active_repair_cycles_20261013.json", json.dumps(snapshot, indent=2) + "\n")
        # Post-fit diagnosis is read-only and runs as real terminal episodes
        # arrive; it cannot silently change training, commands or scoring.
        from VLA.audit_r33_failure_traces import audit_registered
        try:
            audit_registered(fit_root.parent)
        except (OSError, ValueError, KeyError) as error:
            atomic(fit_root.parent/'failure_audit_monitor_error.json', json.dumps({
                'error':str(error),'inference_or_training_interrupted':False},indent=2)+'\n')
        header = ["# Current post-fit evaluation / automation", "",
            f"Updated UTC: {snapshot['updated_at_utc']}; owner alive: {owner}.",
            f"Current stage: `{state.get('stage')}`. Completed standard episodes: {snapshot['completed_standard_trials']}/720.", "",
            f"Actual live cohort: [{cross.name}/REPORT.md]({cross}/REPORT.md).",
            f"Current cycle: [{cycle.name}/REVIEW.md]({cycle}/REVIEW.md).", "",
            "The user now prioritizes MORE causal simulator information. Ordinary completed results remain preserved.",
            "The existing in-flight episode block finishes before its owned inference server closes; no mid-episode cancellation.",
            "Privileged state18 adds target/goal-relative position, requested jaw intent and verified correct grasp; actions remain neural-only.",
            f"Privileged fit logged updates (NOT saved-counter proof): {snapshot['simstate_logged_optimizer_updates_NOT_saved_proof']}; learned trials {snapshot['simstate_completed_trials']}/120.",
            f"Privileged completed120 SR: {snapshot['simstate_SR120']} (pending is NOT 0%).",
            "Other four rows are official ZERO-SHOT diagnostics, not matched fine-tuned paper baselines.",
            "Pending/incomplete is NOT 0%; privileged-input SR is NOT plain VLA SR.", ""]
        completed = neural.get("status") == "completed120_privileged_input_VLA"
        provisional = (f"SmolVLA+SimState COMPLETED DEVELOPMENT120: {snapshot['simstate_provisional_successes']}/120 "
            f"({100*snapshot['simstate_SR120']:.2f}%); NOT an untouched final paper test.") if completed else (
            f"SmolVLA+SimState actual provisional successes: {snapshot['simstate_provisional_successes']}/"
            f"{snapshot['simstate_completed_trials']} completed episodes; NOT completed SR120.")
        rdt_line = (f"RDT+SimState separate cycle: stage `{rdt.get('stage')}`, owned coordinator alive `{rdt.get('owner_alive')}`; "
            f"logged updates (NOT saved-counter proof) {rdt.get('logged_updates_NOT_saved_proof')}/5000; "
            f"learned episodes {rdt.get('completed_trials',0)}/120.") if rdt else "RDT privileged cycle not registered."
        encoder = read(ROOT/"VLA/audits/privileged_robust_encoder40_r28_20261014/progress.json")
        encoder_line = (f"Separate SAME-WEIGHTS input-encoding ablation: {encoder.get('successes',0)}/"
            f"{encoder.get('completed_trials',0)} actual completed, planned40 (10/action); "
            "whole2pi aliases +bounded near-static encoder conditioning, no controller/scorer change.") if encoder else ""
        snapshot["separate_robust_encoder40_NOT_raw120"] = {k:encoder.get(k) for k in ("status","completed_trials","successes","planned_trials")}
        snapshot["other_privileged_native_VLA_queue"] = other
        snapshot["scene64_TRAIN_recovery_data_cycle_NOT_new_SR"] = scene
        snapshot["scene67_full_gate_finetune_cycle"] = fit
        snapshot["other_scene64_full_native_CPU_gates_NOT_finetuning"] = other_inputs
        snapshot["pi0_and_FAST_native32_actual_finetune_queue"] = native32
        snapshot['all_family_privileged_repair_R33_actual_fit_and_test']=repair
        snapshot['OpenPI_shared_EEF_delta_and_native_FAST32_repair']=eef
        snapshot['Smol_scene67_writer_fix_preserve_completed_episodes']=resumed
        snapshot['Smol_native20_matched40_R36_NOT_final_SR']=flow20
        snapshot['Smol_GT_target_crop_motion_balanced_R36']=crop
        snapshot['Smol_native20_conditional_preserve40_extend80_R36']=flow_extension
        fit_probe_root=fit_root.parent/'native_five_family_fit_watch_r37'
        fit_probe_state=read(fit_probe_root/'status.json')
        snapshot['native_five_family_TRAIN_fit_probe_R37_NOT_SR']={
            'cycle':str(fit_probe_root),'owner_alive':alive(fit_probe_state.get('pid',0),fit_probe_root),
            'status':fit_probe_state.get('status'),'models':fit_probe_state.get('models',{}),
            'pending_not_failed':fit_probe_state.get('pending_not_failed',[]),'simulator_controls':0,
            'no_additional_GPU_model':True}
        shield_root=fit_root.parent/'Smol_contactstop_closedloop_r37'
        shield_state=read(shield_root/'status.json');shield_progress=read(shield_root/'progress.json')
        snapshot['Smol_contactstop_closedloop_R37_SEPARATE_additional_safety_shield']={
            'cycle':str(shield_root),'owner_alive':alive(shield_state.get('pid',0),shield_root),
            'stage':shield_state.get('stage'),'status':shield_state.get('status'),'error':shield_state.get('error'),
            'completed_neural_trials':shield_progress.get('completed_trials',0),
            'successes':shield_progress.get('successes'),
            'completed_SR40':shield_progress.get('successes',0)/40 if shield_progress.get('completed_trials')==40 else None,
            'additional_task_independent_safety_projection_NOT_plain_VLA':True}
        bounded_root=fit_root.parent/'Smol_bounded_feedback40_r38'
        bounded_state=read(bounded_root/'status.json');bounded_progress=read(bounded_root/'progress.json')
        bounded={'cycle':str(bounded_root),'owner_alive':alive(bounded_state.get('pid',0),bounded_root),
            'stage':bounded_state.get('stage'),'status':bounded_state.get('status'),'error':bounded_state.get('error'),
            'completed_neural_trials':bounded_progress.get('completed_trials',0),'successes':bounded_progress.get('successes'),
            'completed_SR40':bounded_progress.get('successes',0)/40 if bounded_progress.get('completed_trials')==40 else None,
            'additional_action_shield':False,'separate_untrained_input_bound_ablation':True}
        snapshot['Smol_bounded_feedback_R38_SEPARATE_same_weights_input_ablation']=bounded
        snapshot['Smol_R38_actual_paired_TRAIN_fit_probe_NOT_task_SR']=read(fit_root.parent/'Smol_paired_TRAIN_fit_probe_r38/status.json')
        support=read(fit_root.parent/'Smol_state_support_r38/audit.json')
        snapshot['Smol_R38_observed_state_support_NOT_proven_failure_cause']={
            'TRAIN_frames':support.get('TRAIN',{}).get('frames'),
            'TRAIN_frames_over_bound10':support.get('TRAIN',{}).get('frames_with_any_bound10'),
            'neural_frames':support.get('actual_neural_rollout',{}).get('frames'),
            'neural_frames_over_bound10':support.get('actual_neural_rollout',{}).get('frames_with_any_bound10'),
            'wrong_held_TRAIN_frames':support.get('wrong_held_TRAIN_frames')}
        cohort800={}
        pointer800=read(fit_root.parent/'Smol_800_r39_active.json')
        if pointer800:
            root800=Path(pointer800['cycle'])
            if root800.parent!=ROOT/'VLA/smolvla_scene67' or not root800.name.startswith('test800_200peraction_'):
                raise ValueError('Unexpected fixed800 evaluation owner')
            state800=read(root800/'status.json');progress800=read(root800/'progress.json')
            n800=progress800.get('completed_trials',0);k800=progress800.get('successes',0)
            cohort800={'cycle':str(root800),'owner_alive':alive(state800.get('pid',0),root800),
                'stage':state800.get('stage'),'status':state800.get('status'),'error':state800.get('error'),
                'completed_trials':n800,'planned_trials':800,'planned_per_action':200,'successes':k800,
                'by_action':progress800.get('by_action',{}),'completed_SR800':k800/800 if n800==800 else None,
                'preserved_terminal_trials':progress800.get('preserved_terminal_trials',0),
                'checkpoint_frozen_before_new_outcomes':progress800.get('checkpoint'),
                'new_800_evaluation_only_NOT_training':True,'no_prior120_reused':True}
        snapshot['Smol_R39_frozen_CURRENT_native10_NEW200peraction800']=cohort800
        train_now_root=fit_root.parent/'Smol_actual_TRAIN20_now_r40_v2'
        train_now=read(train_now_root/'status.json')
        snapshot['Smol_R40_original_TRAIN20_actual_fit_probe_NOT_800_or_SR']={
            'cycle':str(train_now_root),'owner_alive':alive(train_now.get('pid',0),train_now_root),
            'status':train_now.get('status'),'stage':train_now.get('stage'),'error':train_now.get('error'),
            'result':read(train_now_root/'actual_TRAIN20/audit.json'),'uses800_outcomes_for_tuning':False}
        chunk_root=fit_root.parent/'Smol_native_chunk2_development40_r41'
        chunk_state=read(chunk_root/'status.json');chunk_progress=read(chunk_root/'progress.json')
        snapshot['Smol_R41_native_chunk2_DEVELOPMENT40_NOT800']={
            'cycle':str(chunk_root),'owner_alive':alive(chunk_state.get('pid',0),chunk_root),
            'status':chunk_state.get('status'),'stage':chunk_state.get('stage'),'error':chunk_state.get('error'),
            'completed_trials':chunk_progress.get('completed_trials',0),'successes':chunk_progress.get('successes'),
            'comparison':read(chunk_root/'matched_comparison.json'),'automatic_promotion':False,
            'frozen800_or_pending_fit_changed':False}
        extended_root=fit_root.parent/'Smol_native_chunk2_conditional120_r41'
        extended_state=read(extended_root/'status.json');extended_progress=read(extended_root/'progress.json')
        snapshot['Smol_R41_native_chunk2_CONDITIONAL_preserve40_extend80']={
            'cycle':str(extended_root),'owner_alive':alive(extended_state.get('pid',0),extended_root),
            'status':extended_state.get('status'),'stage':extended_state.get('stage'),'error':extended_state.get('error'),
            'completed_trials':extended_progress.get('completed_trials',0),'successes':extended_progress.get('successes'),
            'decision':read(extended_root/'conditional_decision.json'),
            'comparison':read(extended_root/'matched_comparison.json'),'NOT_final800_or_new_fit':True}
        snapshot['FAST_actual_codec_offline_physics_quality_NOT_model_SR']=codec
        snapshot['OpenVLA_motion_balanced_measured_recovery_refinement_R34']=balanced
        snapshot["fresh_native_corrective_view_data_ONLY"] = native_recovery
        snapshot["user_render_priority_wait"] = {k:render_wait.get(k) for k in
            ("stage","render_pids","live_render_or_encoder_pids","live_render_count","resumed","updated_at_utc","render_trainer_server_or_episode_signalled")}
        atomic(ROOT / "VLA/audits/active_repair_cycles_20261013.json", json.dumps(snapshot, indent=2) + "\n")
        native_counts = ", ".join(f"{name}: {row.get('neural_controls',0)} actual controls ({row.get('status')})"
            for name,row in other.get("models_native_integration_NOT_task_SR",{}).items())
        other_line = (f"Other native VLA queue: stage `{other.get('stage')}`, owner alive `{other.get('owner_alive')}`; "
            f"native integration NOT task SR: {native_counts}. "
            "Separate OpenVLA branch route2->1k->120; "
            f"actual development episodes {other.get('openvla_privileged_pilot_completed_episodes',0)}/120. "
            f"[Privileged input audit / all-family scope]({other_root}/REPORT.md).") if other_state else ""
        scene_line = (f"New causal scene64 data repair: owner alive `{scene['owner_alive']}`, stage `{scene['stage']}`; "
            f"original source replay {scene['full_original_source64_replayed_episodes']}/2107 episodes, "
            f"{scene['full_original_source64_frames']} frames; "
            f"actual predeclared TRAIN student prefixes {scene['actual_student_TRAIN_prefix_attempts']}/40. "
            f"OFFLINE expert corrections {scene['expert_correction_successes_NOT_VLA_SR']} "
            "are NOT a VLA success rate. New64 model is NOT yet fine-tuned. "
            f"[Audit and next qualification gates]({scene_root}/REPORT.md).") if scene_state else ""
        fit_line = (f"Scene67 corrective finetune: stage `{fit.get('stage')}`, owner alive `{fit.get('owner_alive')}`; "
            "complete physical extraction, real mixed batches, CUDA updates and exact-reload gates precede the2500-update fit. "
            f"New learned development episodes {fit.get('completed_trials',0)}/120; pending is NOT zero success. "
            f"[Deadline / richer-input audit]({fit_root.parent}/REPORT.md).") if fit_state else ""
        native32_line=(f"π0 and FAST actual native32 fit/test queue: stage `{native32.get('stage')}`, "
            f"owner alive `{native32.get('owner_alive')}`; BOTH2 CUDA updates -> exact FP32 reload8 ->1000 updates ->10/action development40. "
            "Still pending is NOT task SR. "
            f"[New input/runtime/data findings]({fit_root.parent}/R32_REVIEW.md).") if native32_state else ""
        if native32.get('orphan_process_audit'):
            native32_line=("Old native-joint OpenPI queue is NOT currently alive despite stale running metadata. "
                "π0 completed40=0/40 remains preserved; FAST1000 completion is NOT established. "
                "Its earlier8-control FAST probe was invalidated:8/8 inner decoder failures silently generated zeros. "
                "The new strict/native32-layout EEF branch is separate; pending is not0%.")
        native_recovery_line=(f"Fresh native corrective observations: stage `{native_recovery.get('stage')}`; "
            f"{native_recovery['collection'].get('completed_attempts',0)}/38 collected offline paths; "
            "own exact replay and real previous handoff cameras required; NOT neural trials or a long-fit qualification.") if native_recovery_state else ""
        render_line=(f"User render priority: `{render_wait.get('stage')}`; "
            f"{len(render_wait.get('live_render_or_encoder_pids',[]))} LIVE render/encoder/parent processes. "
            "Only exact childless VLA waiters are suspended, not trainers, servers, episodes or render jobs; "
            "same PIDs automatically resume after render completion.") if render_wait else ""
        repair_line=(f"NEW all-family privileged queue: `{repair.get('stage')}`; alive `{repair.get('owner_alive')}`. "
            "OpenVLA full64 QLoRA1000, RDT native67+recovery2000, CogACT native cognition/DiT+state-adapter2000; "
            "each requires own CUDA/save/reload8 and30/action development120. "
            f"[Current evidence, variants and results]({fit_root.parent}/R33_REPORT.md).") if repair_state else ""
        eef_line=(f"NEW π0/FAST shared-EEF/recovery queue: `{eef.get('stage')}`; alive `{eef.get('owner_alive')}`. "
            "Own mixed batches ->2 ->exact reload8 ->2000 ->30/action120 EACH. "
            "FAST preserves the official32x32 coefficient layout with only7 physical commands; no silent zeros.") if eef_state else ""
        resumed_line=(f"Smol fitted67 continuation: `{resumed.get('stage')}`; alive `{resumed.get('owner_alive')}`; "
            f"actual completed episodes {resumed.get('development',{}).get('completed_trials',0)}/120. "
            "All5 terminal old trajectories are retained; none rerolled. Resource guard permits only capped inference alongside the exact OpenVLA inference owner, never parallel fitting.") if resumed_state else ""
        balanced_line=(f"OpenVLA R34 motion-balanced recovery refinement: `{balanced.get('stage')}`; alive `{balanced.get('owner_alive')}`; "
            "native batch128 ->own CUDA2/exact reload8 ->1000 new updates ->40 diagnostic episodes. "
            "Other VLA families keep GPU priority; not yet a success improvement. "
            f"[Data exposure and repair]({fit_root.parent}/R34_REVIEW.md).") if balanced_state else ""
        r36_line=(f"R36 native20 learned diagnostic: `{flow20.get('stage')}`, alive `{flow20.get('owner_alive')}`; "
            f"{flow20.get('development',{}).get('successes',0)}/{flow20.get('development',{}).get('completed_trials',0)} completed, planned40 (not final SR). "
            f"GT target-view + numeric67 refinement: `{crop.get('stage')}`, alive `{crop.get('owner_alive')}`; "
            "three-view native128 and full117736 mask frames checked; own CUDA2/reload8 precede2500 updates and120. "
            f"Conditional native20 extension: `{flow_extension.get('stage')}`. [Evidence and unchanged scoring]({fit_root.parent}/R36_REVIEW.md).")
        r37_line=("R37: native20 preserve40/extend80 now permits verified spare capacity of the exact capped RDT trainer, "
            "after actual concurrent reproduction8; no other trainer/episode is interrupted. "
            f"Five-family TRAIN fit probes: `{fit_probe_state.get('status')}`, completed probes `{fit_probe_state.get('models',{})}`. "
            f"These are NOT task SR. [Latest measured work and caveats]({fit_root.parent}/R37_REVIEW.md).")
        header += [provisional,rdt_line,encoder_line,other_line,scene_line,fit_line,native32_line,native_recovery_line,render_line,repair_line,eef_line,resumed_line,balanced_line,r36_line,r37_line,""]
        if progress:
            from VLA.report_cross_vla120 import render
            baseline_report = render(progress)
            # The ordinary queue is explicitly deferred: its active IDE report
            # must show the NEW privileged fit rather than only a stale2.5%.
            if progress.get("status") == "deferred_by_user_privileged_input_priority":
                actual = ["# More-information VLA branch: live status", "",
                    f"Cycle: [{cycle.name}]({cycle}/REVIEW.md); stage `{state.get('stage')}`; owner alive `{owner}`.",
                    "The2.5% below is the COMPLETED original state7/vision row, NOT the result of this new state18 fit.",
                    f"New state18 fit updates (logged estimate, not saved-counter proof): {snapshot['simstate_logged_optimizer_updates_NOT_saved_proof']}/5000.",
                    f"New learned evaluation: {snapshot['simstate_completed_trials']}/120; completed SR120: {snapshot['simstate_SR120']}.",
                    provisional,rdt_line,encoder_line,other_line,scene_line,fit_line,native32_line,native_recovery_line,render_line,repair_line,eef_line,resumed_line,balanced_line,r36_line,
                    "Additional simulator inputs: target-relative pose, goal-relative pose, pad position, previous jaw request, correct bilateral-held feedback.",
                    "No oracle action fallback or scorer/physics change. No promise of high SR before actual learned tests.", "", baseline_report]
                atomic(cross / "REPORT.md", "\n".join(actual))
            header += [baseline_report]
        atomic(ROOT / "VLA/audits/cross_vla120_per_model_20261013/REPORT.md", "\n".join(header))
        if (not owner or state.get("status") != "running") and not any(row.get("owner_alive",False)
            for row in (rdt,other,scene,fit,other_inputs,native32,native_recovery,repair,eef,resumed,codec,balanced,flow20,crop,flow_extension,bounded,cohort800)): return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--replace-owned-monitor",action="store_true",help="Reload only the exact childless evidence writer")
    args = parser.parse_args()
    if args.launch:
        marker = args.cycle / "priority_monitor_launch.json"
        previous=read(marker);history=previous.get("history",[])
        if previous:
            if not args.replace_owned_monitor:raise FileExistsError("Do not duplicate the evidence writer")
            pid=previous["pid"];proc=Path(f"/proc/{pid}")
            if proc.exists() and (proc/"stat").read_text().rsplit(")",1)[1].split()[0]!="Z":
                tokens=(proc/"cmdline").read_bytes().split(b"\0")
                if (b"VLA.monitor_priority_postfit" not in tokens or str(args.cycle.resolve()).encode() not in tokens
                        or (proc/"cwd").resolve()!=ROOT or (proc/"task"/str(pid)/"children").read_text().strip()):
                    raise ValueError("Not the exact childless monitor; no process signalled")
                os.kill(pid,signal.SIGTERM)
                deadline=time.monotonic()+15
                while proc.exists() and (proc/"stat").read_text().rsplit(")",1)[1].split()[0]!="Z":
                    if time.monotonic()>deadline:raise TimeoutError("Owned monitor did not stop")
                    time.sleep(.1)
            history=history+[pid]
        log_path=args.cycle/("priority_monitor.log" if not previous else f"priority_monitor_replacement{len(history)}.log")
        with log_path.open("x") as log:
            p = subprocess.Popen([sys.executable, "-m", "VLA.monitor_priority_postfit", "--cycle", str(args.cycle.resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        atomic(marker, json.dumps({"pid": p.pid, "detached_session": True,"history":history,
            "no_training_or_evaluation_process_signalled":True}) + "\n")
        print(p.pid)
    else:
        monitor(args.cycle.resolve())
