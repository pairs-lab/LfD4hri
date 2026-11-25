"""Evidence-backed captions: measured mechanism versus untested hypothesis."""
import argparse
from pathlib import Path
import numpy as np
from .common import OUT,read
from .ground_context import summarize as ground_summary


def describe(evidence):
    if evidence['full_put_complete']:
        return [f"Complete put: a verified correct hold precedes actual release/support and a continuous 0.5 s stable interval (completion frame {evidence['stable_completion_frame']}).",
            f"Terminal goal-XY error {1000*evidence['terminal_goal_XY_m']:.2f} mm; object speed {evidence['terminal_object_speed_m_per_s']:.6g} m/s.",
            'Incidental contacts earlier in this successful episode are not presented as a failed episode.']
    lines=[]
    for event in sorted(evidence['events'],key=lambda x:x.get('object_displacement_within250ms_m',0),reverse=True)[:3]:
        mode=event['mode']
        if mode=='contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce':
            lines.append(f"At {event['time_s']:.3f} s (frame {event['event_frame']}), actual gripper-target contact {event['force_N']:.2f} N precedes {100*event['object_displacement_within250ms_m']:.2f} cm displacement within 250 ms; peak speed {event['maximum_object_speed_m_per_s']:.2f} m/s. No verified hold at that contact; this is not an in-transit drop by itself.")
        elif mode=='verified_in_transit_drop':
            lines.append(f"A previously verified held object loses attachment at {event['time_s']:.3f} s (frame {event['event_frame']}), {100*event['away_from_goal_XY_m']:.1f} cm from goal, and falls {100*event.get('measured_fall_m',0):.1f} cm. Not a deliberate release at goal.")
        elif mode=='empty_gripper_lift_after_failed_acquisition_no_object_lift':
            lines.append(f"After an unsuccessful near-object closing attempt (frame {event['event_frame']}), TCP rises {100*event['TCP_lift_m']:.1f} cm while the requested object moves at most {1000*event['target_displacement_m']:.2f} mm. Verified hold never occurs. This directly shows an empty-hand continuation, not object release/drop.")
            if 'object_vertical_lift_during_empty_lift_m' in event:
                lines.append(f"The requested object's vertical rise during that empty lift is only{1000*event['object_vertical_lift_during_empty_lift_m']:.3f}mm; pad-object vertical separation reaches{100*event['vertical_pad_object_separation_at_lift_m']:.2f}cm. Incidental lateral sliding is not a successful grasp.")
        elif mode=='clear_planar_acquisition_miss_empty_closure_no_recovery':
            lines.append(f"The minimum pad-target XY separation throughout the rollout is{100*event['minimum_pad_target_XY_over_whole_rollout_m']:.1f}cm, despite{event['requested_close_controls']} closing controls. The target moves only{1000*event['target_displacement_m']:.3f}mm and is never verified held. Observed localization/acquisition miss, not proof of identity misunderstanding.")
        elif mode=='controlled_Oracle_transport_timeout':
            lines.append('The explicitly speed-limited Oracle reaches the shared22s physical-time budget while transport is incomplete. This is not slow neural inference or an unmodified learned-RL weakness.')
        elif mode=='controlled_Oracle_failed_acquisition_empty_closure':
            lines.append('The explicitly perturbed Oracle closes/continues at an unsuccessful acquisition pose and never obtains a verified grasp. Do not describe this as a slam or bounce unless separate contact/velocity evidence establishes it.')
        else:lines.append('Observed event: '+mode+'. See content_evidence.json for actual frames and measurements.')
    lines.append(f"No completed put: terminal goal-XY error{100*evidence['terminal_goal_XY_m']:.2f}cm; verified hold frames{evidence['held_frames']}.")
    return lines


def main(a):
    pair=read(a.pair/'PAIR_AUDIT.json')
    task=read(a.pair/'XVLA_terminal_release/initial_scene.json')['task']
    label='Backup' if pair['backup'] else 'Scene'
    lines=[f"# {label}{pair['scene_id']:02d}: {task['prompt']}",'',
        'Outcome-selected qualitative illustration, not a quantitative success-rate estimate.','',
        '## Matched task and disclosed method variants','',
        f"Requested object: {task['pick_name']}; destination: {task['place_name']}. Exact measured initial-world SHA: `{pair['initial_state_sha256']}`.",
        'The folder alias RL means privileged feedback Oracle, NOT a trained RL policy. Any injected controller fault is disclosed below.',
        'X-VLA + simulator GT uses genuine frozen step014500 neural approach/carry actions with text, native proprioception,67 privileged numeric measurements and THREE actual visual inputs: original front RGB, original wrist RGB, and a deterministic GT diagram of target/goal/pad geometry, height and contact/holding flags. It is not vanilla two-camera X-VLA. Only terminal holding/opening is a separately disclosed non-neural supervisor; no arm waypoint correction.',
        'Both use the shared body-contact Robotiq plant and bilateral-contact-triggered grasp assist. Oracle and neural-policy motor interfaces are different controllers; do not call this identical action-generation logic.',
        'Camera recording is read-only. Policy/physics use the pinned original binding; the alternate non-stepping image binding is checked against lossless RGB/segmentation goldens.','']
    seed=read(a.pair/'XVLA_terminal_release/initial_scene.json')['seed']
    input_dir=f"../../audit/policy_inputs/{label.lower()}{pair['scene_id']:02d}_seed{seed}"
    lines+=['Actual policy input audit: [reset front RGB]('+input_dir+'/reset_front.png), '
        '[reset wrist RGB]('+input_dir+'/reset_wrist.png), '
        '[all original-query GT diagrams]('+input_dir+'/actual_NN_map_queries.png). '
        'Reconstructed GT-image bytes are matched to ORIGINAL actual-server receipts; complete30-action chunks are also hash-verified. Partial terminal chunks are not claimed full-hash-verified. No new NN inference is performed.','']
    for method in ('Scripted_Oracle','XVLA_terminal_release'):
        folder=a.pair/method;e=read(folder/'content_evidence.json');lines+=['## '+method,'']
        lines+=['- '+text for text in describe(e)]
        if method=='Scripted_Oracle':
            o=read(folder/'outcome.json')
            lines+=['',f"Explicit Oracle fault: `{o['fault_script']}`. Recorded fault events: `{o['fault_events']}`."]
        else:
            proof=read(folder/'replay_proof.json')
            pixels=[r['actual_policy_visibility']['wrist']['target_pixels'] for r in proof['controls']]
            absent=sum(v==0 for v in pixels)
            lines+=['',f'Actual policy-wrist target-absent controls: {absent}/{len(pixels)}. This is NOT the presentation camera and does not establish memory loss; privileged state remains supplied at native neural queries.']
            if not e['full_put_complete']:
                assignment=next(r for r in read(OUT/'assignments.json') if
                    r['scene_id']==pair['scene_id'] and r['backup']==pair['backup'])
                contact=ground_summary(assignment['vla'])
                if contact['events']:
                    lines+=['','Surface-contact counterevidence (dense240Hz telemetry):']
                    for event in contact['events']:
                        velocity=event['entry_vertical_velocity_50ms_m_per_s']
                        lines.append(f"- Ground contact starts{event['first_contact_time_s']:.3f}s, peak solver force{event['peak_force_N']:.1f}N; preceding50ms TCP vertical velocity{velocity:.3f}m/s. High force alone is NOT a violent downward slam. Grasp height, finger/table interference and low-level contact control remain competing explanations for acquisition failure; this clip does not isolate a semantic/localization defect.")
                    lines.append('These are simulator solver forces, not calibrated real-robot impact-force estimates.')
        lines+=['']
    lines+=['## Counterarguments / limits','',
        '- A miss, empty-hand lift or impact is observable; model belief, intrinsic architecture limits, memory loss and semantic misidentification are not established by these videos.',
        '- Native action chunks, trained data and the shared controller/guard affect behavior. Treat this as the tested policy/protocol, not a ceiling for all VLA methods.',
        '- Artificial Oracle controller faults illustrate specified mechanisms. They do not establish natural failure rates or robustness of trained RL.',
        '- Complete-episode videos are uncut at30FPS physical time. Full-stream decode/cadence/format and sampled actual encoded-event inspection are separate gates. Manual continuous every-frame viewing is not claimed.',
        '- Backup aliases and additional camera views are not extra independent trials. No selected qualitative count is added to quantitative SR.','']
    (a.pair/'EXPLANATION.md').write_text('\n'.join(lines))
    index=a.pair/'index.html';page=index.read_text().replace('X-VLA + terminal release','X-VLA + simulator GT + terminal release')
    if 'EXPLANATION.md' not in page:page+='<p><a href="EXPLANATION.md">Measured failure mechanisms, actual success and counterarguments</a></p>'
    index.write_text(page)
    for method,entry in pair['outcomes'].items():
        alias=Path(entry['delivery_alias']);text=alias.with_suffix('.txt').read_text()
        text='\n'.join(line for line in text.splitlines() if not line.startswith('Measured mechanism, counterarguments and shared-assist/controller disclosures:'))
        alias.with_suffix('.txt').write_text(text+'\nMeasured mechanism, counterarguments and shared-assist/controller disclosures: '+str(a.pair/'EXPLANATION.md')+'\n')
    print({'explanation':str(a.pair/'EXPLANATION.md')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,required=True)
    main(parser.parse_args())
