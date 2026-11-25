"""Freeze the targeted put-only scene/fault proposals before paired outcomes."""
import argparse
from .common import OUT,read,tasks
from VLA.common.artifact_io import atomic_json

CASES=[
    (1,20269719,'none','01_RL_succeed_VLA_succeed'),
    (2,20269711,'none','01_RL_succeed_VLA_succeed'),
    (3,20269727,'drop_in_transit','02_RL_fail_VLA_succeed'),
    (4,20268104,'approach_overshoot','02_RL_fail_VLA_succeed'),
    (5,20269701,'slow_transport_timeout','02_RL_fail_VLA_succeed'),
    (6,20269730,'none','03_RL_succeed_VLA_fail'),
    (7,20269722,'none','03_RL_succeed_VLA_fail'),
    (8,20269703,'none','03_RL_succeed_VLA_fail'),
    (9,20269718,'drop_in_transit','04_RL_fail_VLA_fail'),
    (10,20268017,'approach_overshoot','04_RL_fail_VLA_fail'),
    (11,20269704,'drop_in_transit','04_RL_fail_VLA_fail'),
    (None,20269721,'none','05_backups'),
    (None,20269705,'drop_in_transit','05_backups'),
    (None,20269715,'slow_transport_timeout','05_backups'),
    (None,20268016,'approach_overshoot','05_backups'),
]


def main():
    rows=read(OUT/'audit/initial_trace_triage/audit.json')['rows']
    by_seed={r['seed']:r for r in rows};cases=[]
    for scene_id,seed,fault,group in CASES:
        row=by_seed[seed];task=tasks(seed)[-1]
        cases.append({**row,'proposed_scene_id':scene_id,'proposed_group':group,'oracle_fault':fault,
            'canonical_task':task.to_dict(),'status':'PROPOSED_NOT_outcome_certified',
            'no_VLA_scripted_failure':True,'hybrid_VLA_terminal_release_enabled':True,
            'Oracle_fault_disclosed_NOT_native_learned_RL_failure':fault!='none'})
    atomic_json(OUT/'candidate_plan.json',{'scope':'put_only_qualitative_TARGETED_NOT_quantitative_SR',
        'candidate_pairs':cases,'desired_main_pairs':11,'backups':4,
        'all11_desired_outcomes_are_NOT_claimed_before_tests':True,
        'select_VLA_failures_before_terminal_release_NOT_trivial_release_miss':True})
    atomic_json(OUT/'all_trace_candidates.json',rows)
    print({'predeclared_candidates':len(cases),'put_only':True,'main_proposals':11,'backups':4},flush=True)


if __name__=='__main__':main()
