"""Predeclared measurable shortlist gates; never replaces actual video review.

The selected sample is qualitative and outcome-conditioned, not an SR estimate
or proof of model-intrinsic weakness. Safety/controller-only failures are kept
in backups. Weak releases and mere loss of camera visibility are insufficient.
"""


def assess_vla(evidence, proof):
    if evidence['full_put_complete']:
        return {'eligible_for_render':True,'strong_failure_candidate':False,
            'reason':'verified_complete_put_NOT_a_failure','accepted_events':[],
            'manual_encoded_visual_review_required':True}
    rejection=[]
    if proof.get('benchmark_region_success'):
        rejection.append('Only_incomplete_or_rejected_terminal_release_not_requested_strong_failure')
    if proof.get('safety_violation'):
        rejection.append('Safety_or_mechanical_abort_needs_system_attribution_not_VLA_weakness_claim')
    accepted=[]
    for event in evidence.get('events',[]):
        mode=event['mode']
        if mode=='contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce':
            if (event['force_N']>=1. and event['object_displacement_within250ms_m']>=.08
                    and event['maximum_object_speed_m_per_s']>=.5):
                accepted.append(event)
        elif mode=='verified_in_transit_drop':
            if event['actual_correct_hold_before_drop'] and event.get('measured_fall_m',0)>=.03:
                accepted.append(event)
    if not accepted:
        rejection.append('No_strong_measured_pre_release_impact_or_verified_drop_yet')
    return {'eligible_for_render':not rejection,'strong_failure_candidate':not rejection,
        'reason':'measurable_pre_release_failure_PENDING_visual_review' if not rejection else '; '.join(rejection),
        'accepted_events':accepted,'rejected_as_sufficient_alone':[
            'off_goal_release','target_absent_in_wrist','clearance_guard_stall',
            'safety_abort_without_actual_contact','unproven_memory_or_identity_claim'],
        'qualitative_selection_NOT_quantitative_SR':True,
        'causation_or_superiority_over_RL_NOT_established':True,
        'manual_encoded_visual_review_required':True}
