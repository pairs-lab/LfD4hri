from VLA.qualitative_r57.selection import assess_vla as previous


def assess(evidence,proof):
    result=previous(evidence,proof)
    if (not evidence['full_put_complete'] and not proof['benchmark_region_success']
            and not proof['safety_violation']):
        extra=[e for e in evidence['events'] if e['mode'] in (
            'clear_planar_acquisition_miss_empty_closure_no_recovery',
            'empty_gripper_lift_after_failed_acquisition_no_object_lift')]
        if extra:
            result.update(eligible_for_render=True,strong_failure_candidate=True,
                reason='measured_pre_release_failed_acquisition_PENDING_actual_visual_review',
                accepted_events=result['accepted_events']+extra)
    return result
