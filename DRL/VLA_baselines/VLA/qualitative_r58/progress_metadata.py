"""Correct inherited recorder progress labels only after owned outputs close."""
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,read


def main():
    corrections=[]
    for marker in (OUT/'_renders').glob('*/render_complete.json'):
        directory=marker.parent;path=directory/'recording_progress.json'
        if not path.exists():continue  # Do not alter reused original-source files.
        before=sha256_file(path);old=read(path);frames=read(directory/'frames.json')
        neural=(directory/'replay_proof.json').exists()
        tail=read(directory/'replay_proof.json').get('terminal_release_supervisor') if neural else None
        pure_neural=neural and not bool((tail or {}).get('controls'))
        new={**old,'status':'closed_render_verified','frames':len(frames),
            'physics_tick':frames[-1]['physics_tick'],'simulation_seconds':frames[-1]['simulation_time_s'],
            'all_actions_from_recorded_neural_trace':pure_neural,
            'recorded_control_origin':('VLA_native_plus_disclosed_terminal_supervisor' if neural and not pure_neural
                else 'VLA_native_neural_actions' if neural else 'scripted_feedback_Oracle_NOT_neural'),
            'legacy_generic_neural_progress_label_corrected_without_video_or_action_change':True}
        atomic_json(path,new)
        corrections.append({'file':str(path.relative_to(OUT)),'before_sha256':before,'after_sha256':sha256_file(path),
            'only_private_closed_progress_metadata_changed':True})
    atomic_json(OUT/'audit/closed_progress_metadata_corrections.json',corrections)
    print({'closed_progress_labels_corrected':len(corrections),'videos_outcomes_actions_unmodified':True},flush=True)


if __name__=='__main__':main()
