"""Apply the user's narrower view request without losing completed PNG work."""
from pathlib import Path
import shutil
from .export import OUTPUT,read,write,sha,lock,utc


def main():
    out=OUTPUT
    with lock(out):
        old=read(out/'INVENTORY.json');pilot=read(out/'PILOT_AND_ADMISSION.json')
        assert old['stream_count']==294 and old['frames']==104608
        archive=out.parent/'qualitative_rl_vla_superseded_all_views_20261018'
        archive.mkdir(exist_ok=False)
        code=archive/'extractor_sources';code.mkdir()
        manifest=read(out/'EXTRACTION_SOURCE_MANIFEST.json')
        for path,digest in manifest['sources'].items():
            source=Path(path)
            assert sha(source)==digest, 'Preserve the original code before rescoping it'
            shutil.copy2(source,code/source.name)
        selected={'view2','view3','view4','view6'}
        rows=[dict(r) for r in old['streams'] if r['camera'] in selected]
        seen={}
        for row in rows:
            row['duplicate_of_stream']=seen.get(row['source_sha256'])
            seen.setdefault(row['source_sha256'],row['id'])
        data={**old,'UTC':utc(),'streams':rows,'stream_count':len(rows),
              'unique_source_videos':len(seen),'frames':sum(r['frames'] for r in rows),
              'source_bytes_all_folder_copies':sum(r['source_bytes'] for r in rows),
              'selected_wide_views':[2,3,4,6],
              'native_handeye_exported':False,
              'scope_amendment':'User requested only views 2, 3, 4, 6; no new robot runs',
              'previous_full_view_inventory':str(archive/'INVENTORY.json')}
        moved=[]
        for row in old['streams']:
            parent=out/row['output_method']
            candidates=[]
            if row['camera'] not in selected:
                candidates=[out/row['output_frames'],parent/(row['source_camera']+'.mp4')]
                candidates += [parent/(row['camera']+suffix) for suffix in
                               ('.audit.json','.sha256','.timestamps.tsv','.ffmpeg.log')]
            candidates += list(parent.glob('.'+row['camera']+'.extracting_*'))
            for path in candidates:
                if not path.exists():continue
                dest=archive/path.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True)
                assert not dest.exists();path.rename(dest);moved.append(str(path.relative_to(out)))
        for path in list(out.iterdir()):
            if path.is_file() and path.name not in ('.export.lock','.automatic.lock'):
                dest=archive/path.name
                assert not dest.exists();path.rename(dest);moved.append(path.name)
        if (out/'_pilot').exists():(out/'_pilot').rename(archive/'_pilot')
        samples={h:pilot['samples'][h] for h in seen}
        estimate=sum(samples[r['source_sha256']]['average_bytes']*r['frames'] for r in rows)
        conservative=sum(samples[r['source_sha256']]['max_bytes']*r['frames'] for r in rows)*1.30
        required=int(conservative+data['source_bytes_all_folder_copies']+512*1024**2)
        free=shutil.disk_usage(out).free
        assert free-required>=25*1024**3
        new_pilot={**pilot,'sampled_unique_videos':len(samples),'samples':samples,
                   'expected_PNG_GiB':estimate/1024**3,'conservative_PNG_GiB':conservative/1024**3,
                   'required_conservative_GiB':required/1024**3,
                   'free_GiB_before_resumed_export':free/1024**3,
                   'selected_wide_views':[2,3,4,6],
                   'pilot_reused_from':str(archive/'PILOT_AND_ADMISSION.json'),
                   'pilot_scope':'All selected videos already passed original PNG/RGB pilot'}
        write(out/'INVENTORY.json',data);write(out/'PILOT_AND_ADMISSION.json',new_pilot)
        write(out/'SCOPE_AMENDMENT.json',{'UTC':utc(),'selected_wide_views':[2,3,4,6],
             'requested_groups':old['groups'],'all_pairs_and_both_methods_preserved':True,
             'preserved_previous_attempt':str(archive),'moved_files_or_directories':moved,
             'old_export_process_group_stopped':1624571,'no_source_videos_deleted_or_changed':True,
             'completed_selected_PNGs_kept_in_place':True,'source_code_before_amendment':str(code)})
        write(out/'PROGRESS.json',{'status':'scope_changed_to_views_2_3_4_6_ready_to_resume',
              'streams_total':len(rows),'frames_total':data['frames'],'UTC':utc()})
        print({'selected_views':[2,3,4,6],'streams':len(rows),'frames':data['frames'],
               'unique_videos':len(seen),'archive':str(archive)},flush=True)


if __name__=='__main__':main()
