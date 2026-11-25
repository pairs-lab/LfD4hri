"""Verify published counts, labels, chosen paired views and untouched frames."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import decode_at
from VLA.qualitative_r57.common import read
from .prepare import OUT,OLD


def main():
    rows=read(OUT/'LABELS.json');assert len(rows)==34
    aliases=[]
    for row in rows:
        video=OUT/row['video'];assert sha256_file(video)==row['sha256']
        assert 1<=len(row['label'].split())<=2
        assert row['instruction'] and row['requested_object'] and row['destination']
        if row.get('recommended_video'):
            best=OUT/row['recommended_video'];assert sha256_file(best)==row['recommended_video_sha256']
            method='Scripted_Oracle' if row['method_alias']=='RL' else 'XVLA_terminal_release'
            expected=Path(row['paired_scene'])/method/f'view_{row["recommended_fixed_camera"]}.mp4'
            assert sha256_file(expected)==sha256_file(best)
        aliases.append(str(video))
    main=[r for r in rows if not r['backup']];assert len(main)==26
    groups=Counter(Path(r['video']).parts[0] for r in main if r['method_alias']=='RL')
    assert sorted(groups.values())==[2,3,4,4]
    assert sum(len(list((OUT/g).glob('*.mp4'))) for g in groups)==26
    assert len(list((OUT/'05_backups').glob('*.mp4')))==8
    for item in read(OUT/'produced_pairs.json'):
        pair=Path(item['paired_output']);audit=read(pair/'PAIR_AUDIT.json')
        assert audit['same_measured_initial_state'] and audit['Oracle_repeat_proof']['passed']
        assert read(pair/'FINAL_VISUAL_REVIEW.json')['approved_for_delivery']
        assert read(pair/'review/independent/actual_visual_review.json')['passed']
        for method in ('Scripted_Oracle','XVLA_terminal_release'):
            a=read(pair/method/'camera_audit.json')
            assert a['media_decode_sync_pass'] and a['actual_tick_spacing_pass']
            assert len(a['streams'])==7
    frame_rows=read(OUT/'HIGHLIGHT_FRAMES/FRAME_MANIFEST.json')
    for row in frame_rows:
        file=OUT/'HIGHLIGHT_FRAMES'/row['file']
        assert sha256_file(file)==row['PNG_sha256']
        decoded=decode_at(row['source_video'],row['frame_index_0_based'])
        assert np.array_equal(cv2.imread(str(file)),decoded)
    immutable=read(OUT/'audit/original_integrity/independent_delivery_integrity.json')
    assert immutable['passed'] and sha256_file(OLD/'MANIFEST.sha256.json')==immutable['original_manifest_sha256']
    result={'passed':True,'verified_UTC':datetime.now(timezone.utc).isoformat(),
        'main_videos':26,'backup_videos':8,'main_pairs':13,'main_group_pair_counts':dict(groups),
        'all_named_alias_hashes_verified':True,'all_labels_one_or_two_words':True,
        'new_pairs_same_initial_world_and_Oracle_repeat_exact':True,
        'new_pairs_whole_stream_media_pass':True,'new_pairs_actual_sampled_visual_review_pass':True,
        'recommended_view_same_for_both_methods_in_each_new_scene':True,
        'exported_highlight_frames_pixel_exact_vs_encoded_source':len(frame_rows),
        'original_2207_entry_integrity_audit_pass':True,
        'Oracle_is_STAGED_scripted_controller_NOT_learned_RL':True,
        'scene12_native_VLA_reused_from_scene01_NOT_an_extra_independent_trial':True,
        'semantic_misidentification_and_memory_loss_NOT_proven':True,
        'no_continuous_every_frame_human_playback_claim':True}
    atomic_json(OUT/'FINAL_DELIVERY_AUDIT.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
