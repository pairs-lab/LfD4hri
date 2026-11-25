"""Independent delivery checks; never infers scientific meaning from aesthetics.

Run after --stage assemble, with a separately written honest visual_review.json.
Checks all source/panel hashes, PDF font embedding, editable SVG text, PNG DPI,
frame linkage, and the actual policy-input provenance. Does not train or simulate.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .qualitative_figure import OUTPUT,ROOT,read,write,digest


def main(args):
    out=args.output.resolve();manifest=read(out/'manifest.json');audit=read(out/'audit.json')
    visual=read(out/'visual_review.json')
    if not audit['automated_pass'] or audit['stage']!='full':raise ValueError('Full automatic gate not passed')
    if set(visual['reviewed_case_ids'])!=set(manifest['case_ids']) or not visual['all_cases_accepted']:
        raise ValueError('Missing or failed visual review')
    alignment=read(out/'policy_alignment.json')
    if not alignment['verified'] or not visual.get('HD_to_query_alignment_reviewed'):
        raise ValueError('HD-to-policy-query alignment not reviewed')
    align_map={r['HD_frame_index_0_based']:r for r in alignment['records']}
    if not any('scene13_HD_query_alignment.pdf' in f['files'][0] for f in audit['figures']):
        audit['figures'].append(alignment['figure'])
    manifest['policy_HD_query_alignment']=alignment['records']
    tests=subprocess.run([sys.executable,'-m','unittest','env_eval.case_demon.test_qualitative_figure','-v'],
        cwd=ROOT,capture_output=True,text=True)
    (out/'unit_tests.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise ValueError('Unit test regression: see unit_tests.txt')
    figure_rows=[]
    for figure in audit['figures']:
        if figure['minimum_figure_font_pt']<8 or figure['text_outside_page']:
            raise ValueError('Font size or page overflow failure')
        paths={Path(p).suffix:out/p for p in figure['files']}
        with Image.open(paths['.png']) as image:
            dpi=image.info.get('dpi',(0,0))
            expected=np.asarray([figure['width_mm'],figure['height_mm']])*600/25.4
            if max(abs(np.asarray(dpi)-600))>.1 or max(abs(np.asarray(image.size)-expected))>2:
                raise ValueError('PNG size/density mismatch')
            dimensions=list(image.size)
        tree=ET.parse(paths['.svg']);root=tree.getroot();ns={'s':'http://www.w3.org/2000/svg'}
        texts=root.findall('.//s:text',ns);embedded=root.findall('.//s:image',ns)
        if not texts or not embedded:raise ValueError('SVG lost editable text or source images')
        for item in embedded:
            href=item.get('{http://www.w3.org/1999/xlink}href',item.get('href',''))
            if not href.startswith('data:image/'):
                raise ValueError('SVG source image is not self-contained')
        fonts=subprocess.run(['pdffonts',str(paths['.pdf'])],capture_output=True,text=True,check=True).stdout
        font_rows=[line.split() for line in fonts.splitlines()[2:] if line.strip()]
        if not font_rows or any('yes' not in line for line in font_rows):
            raise ValueError('PDF text fonts not embedded')
        info=subprocess.run(['pdfinfo',str(paths['.pdf'])],capture_output=True,text=True,check=True).stdout
        if 'Pages:           1' not in info:raise ValueError('Unexpected PDF page count')
        figure_rows.append({'file':str(paths['.pdf'].relative_to(out)),
            'PDF_fonts_embedded':True,'SVG_text_nodes':len(texts),'SVG_raster_image_nodes':len(embedded),
            'PNG_pixels':dimensions,'PNG_dpi':list(dpi),
            'native_width_mm':figure['width_mm'],'native_height_mm':figure['height_mm'],
            'minimum_text_pt_at_native_width':figure['minimum_figure_font_pt']})
    source_count=0;panel_count=0;decode_frames=0;csv_rows=[]
    for case in manifest['cases']:
        times_seen=[]
        for method in case['methods']:
            outcome_path=Path(method['published'])/'outcome.json'
            if outcome_path.exists():
                outcome=read(outcome_path)
                method['outcome_provenance']={'file':str(outcome_path),'sha256':digest(str(outcome_path)),
                    'fault_script':outcome.get('fault_script'),'fault_events':outcome.get('fault_events',[]),
                    'error':outcome.get('error')}
            for path,expected in method['source_sha256'].items():
                if digest(path)!=expected:raise ValueError('Original source modified: '+path)
                source_count+=1
            for role,check in method['decode_audit'].items():
                if check.get('minimum_spatial_channel_std',0)<1.:
                    raise ValueError('Source was not checked for spatially empty RGB')
                decode_frames+=check['decoded_frames']
            frames=read(Path(method['clean'])/'frames.json')
            labels={q['index']:q['label'] for q in method['selection']['main']}
            previous=-1.
            for row in method['panels']:
                i=row['frame_index_0_based'];actual=frames[i];t=row['simulation_time_s']
                if t<=previous or abs(t-actual['simulation_time_s'])>1e-10 or abs(t-row['physics_tick']/240)>1e-10:
                    raise ValueError('Frame/time linkage failed')
                if row['common_pair_crop_xyxy']!=case['crop'] or row['camera_view']!=case['view']:
                    raise ValueError('Paired camera/crop changed')
                previous=t
                for item in row['files'].values():
                    path=out/item['file']
                    if digest(str(path))!=item['sha256']:raise ValueError('Original panel was modified')
                    with Image.open(path) as image:
                        signal=float(np.asarray(image).astype(np.float32).std(axis=(0,1)).max())
                        if signal<1.:raise ValueError('Blank selected panel')
                    panel_count+=1
                csv_rows.append({'scene':case['scene_id'],'seed':case['seed'],'method':method['name'],
                    'checkpoint':case['shared_XVLA_checkpoint'] if method['name']!='Scripted_Oracle' else 'scripted',
                    'camera':case['view'],'frame_index_0_based':i,'simulation_time_s':t,
                    'encoded_PTS_s':row['encoded_PTS_s'],'event':labels.get(i,'detail'),
                    'latest_policy_query_s':align_map[i]['latest_policy_query_time_s']
                        if case['scene_id']==13 and method['name']=='XVLA_terminal_release' and i in align_map else '',
                    'workspace_crop':row['files']['workspace_crop']['file'],
                    'handeye_HD':row['files']['handeye']['file'],
                    'outcome':method['selection']['outcome']})
            times_seen.append([r['simulation_time_s'] for r in method['panels']])
    if len(manifest['policy_observations'])!=10:raise ValueError('Expected 5 paired RGB queries')
    for item in manifest['policy_observations']:
        if digest(str(out/item['file']))!=item['sha256'] or digest(item['original'])!=item['sha256']:
            raise ValueError('Actual RGB policy observation mismatch')
    write(out/'manifest.json',manifest)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    (out/'frame_selection.csv').write_text(buffer.getvalue())
    write(out/'delivery_checks.json',{'passed':True,'source_hash_rechecks':source_count,
        'unannotated_panel_hash_and_nonblank_checks':panel_count,
        'decoded_video_frames_across_both_cameras':decode_frames,'figures':figure_rows,
        'manual_review_record_sha256':digest(str(out/'visual_review.json')),
        'unit_tests_passed':True,'unit_test_log':'unit_tests.txt',
        'every_video_frame_human_viewing_claimed':False})
    audit.update(completed_delivery=True,manual_visual_review_required=False,
        manual_visual_review_completed=True,delivery_checks='delivery_checks.json')
    write(out/'audit.json',audit)
    write(out/'status.json',{'stage':'complete','completed_delivery':True,'cases':len(manifest['cases']),
        'main_case_ids':manifest['main_ids'],'figures':len(figure_rows),'manual_and_automatic_audits_passed':True})
    write(out/'evidence_limits.json',{
        'Oracle_is_scripted_not_trained_TD3':True,
        'Oracle_failure_illustrations_are_staged':True,
        'XVLA_uses_simulator_privileged_inputs_shared_servo_grasp_and_safety':True,
        'terminal_release_opening_only_supervisor_disclosed':True,
        'camera_HD_is_not_actor_input':True,
        'case13_policy_inputs_are_actual_RPC_verified_RGB224_queries':True,
        'no_memory_loss_or_semantic_misidentification_causal_claim':True,
        'qualitative_selection_is_post_hoc_not_failure_frequency_or_new_SR':True,
        'qualitative_full_put_gate_is_not_strict_1cm_benchmark_success':True,
        'source_full_put_classifier':{'maximum_XY_error_m':0.045,'maximum_object_speed_m_per_s':0.025,
            'minimum_support_force_N':0.005,'minimum_continuous_settle_s':0.5,
            'requires_actual_prior_hold_and_terminal_opening':True},
        'scene12_reuses_published_XVLA_source':True,
        'scene14_archived_neural_drop_then_reacquisition_has_no_induced_failure':True,
        'new_neural_rollouts':0,'new_Oracle_no_fault_counterparts':1,
        'original_quantitative_results_unchanged':True,
        'six_frames_in_detail_do_not_show_every_in_between_state':True,
        '600_DPI_does_not_invent_source_detail':True,
        'print_warning':'Main is 180 mm wide but taller than an ordinary manuscript text page. Use print_halves at native 180 mm width. Detail pages are 270 mm wide; shrinking them to 180 mm also shrinks 8pt labels.',
        'rejected_render':'v1 recovery rendering was blank because EGL loaded after reset. v2 fixed initialization and enforces nonblank checks. v1 is not a final source.'})
    supplement=(' Qualitative placement completion uses the archived release/support/settle gate; '
        'it must not be read as strict 1 cm benchmark success.')
    caption=(out/'caption.txt').read_text().strip()
    if supplement.strip() not in caption:
        (out/'caption.txt').write_text(caption+supplement+'\n')
        page=(out/'index.html').read_text()
        import html
        page=page.replace(html.escape(caption),html.escape(caption+supplement))
        (out/'index.html').write_text(page)
    commands=[
        'python -m env_eval.case_demon.qualitative_figure --stage sources --include-recovery',
        'python -m env_eval.case_demon.qualitative_figure --stage figures --include-recovery --only main_6_cases,print_halves/main_partA,print_halves/main_partB,groups/group01,policy_observations/scene13_actual_policy_inputs',
        'python -m env_eval.case_demon.qualitative_figure --stage figures --include-recovery --only groups/group02,groups/group03,groups/group04',
        'python -m env_eval.case_demon.qualitative_figure --stage figures --include-recovery --only details/scene02,details/scene14,details/scene03,details/scene04',
        'python -m env_eval.case_demon.qualitative_figure --stage figures --include-recovery --only details/scene05,details/scene12,details/scene06,details/scene07',
        'python -m env_eval.case_demon.qualitative_figure --stage figures --include-recovery --only details/scene08,details/scene13,details/scene10,details/scene11',
        'python -m env_eval.case_demon.qualitative_figure --stage assemble --include-recovery',
        'python -m env_eval.case_demon.qualitative_policy_alignment',
        'python -m unittest env_eval.case_demon.test_qualitative_figure -v',
        'python -m env_eval.case_demon.qualitative_delivery_audit']
    (out/'REPRODUCE.txt').write_text('Run from '+str(ROOT)+'. No training or neural inference.\n'
        'The additional recovery capture and v2 render are already preserved under additional_recovery_source.\n'
        'Review output images honestly and update visual_review.json before the final audit.\n\n'+'\n'.join(commands)+'\n')
    (out/'README.txt').write_text(
        'Open index.html for the complete gallery.\n'
        'main_6_cases.pdf/svg/png: six paired cases, all four outcome groups.\n'
        'print_halves/: recommended 180 mm-wide manuscript-size parts A/B.\n'
        'groups/: four complete outcome-group pages, twelve paired cases total.\n'
        'details/: twelve six-timepoint-per-method detail pages, native width 270 mm.\n'
        'policy_observations/: actual RGB224 at five recorded queries plus explicit HD-to-query alignment.\n'
        'panels/: unannotated original wide/handeye images and identical pair crops.\n'
        'review/: event-dense review sheets, NOT replacement source imagery.\n'
        'caption.txt: English scientific caption with assistance/selection disclosures.\n'
        'manifest.json and frame_selection.csv: source SHA-256, exact frames/times/camera/crop/events.\n'
        'audit.json, delivery_checks.json, visual_review.json, evidence_limits.json: completed checks and limits.\n'
        'Scene14 replaces a redundant Scene01; the main overview replaces Scene02 with Scene14 to show natural recovery. Scene02 remains in the complete set.\n'
        'Only Scripted Oracle and X-VLA are shown, not a trained TD3 comparison.\n'
        'No model retraining, modified VLA actions, forced VLA failures, AI-generated robot images, or changed quantitative results.\n')
    import html
    rows=[]
    for case in manifest['cases']:
        notes=visual['case_notes'][str(case['scene_id'])]
        outcomes=' / '.join(m['selection']['outcome'] for m in case['methods'])
        distances=' / '.join(f"{100*m['evidence']['terminal_goal_XY_m']:.2f} cm" for m in case['methods'])
        rows.append('<tr><td><a href="details/scene%02d.pdf">S%02d</a></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            case['scene_id'],case['scene_id'],html.escape(case['short_instruction']),
            html.escape(outcomes),html.escape(distances),html.escape(notes)))
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Qualitative comparison: audit report</title>
<style>body{max-width:1300px;margin:32px auto;padding:0 24px;font:15px/1.55 system-ui;color:#18222c}
table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #d1d7dc;text-align:left;vertical-align:top}
.note{padding:16px;background:#eef3f7}a{color:#0072b2}</style><h1>Evidence and delivery audit</h1>
<p><a href="index.html">Figure gallery</a> · <a href="manifest.json">Exact source/frame manifest</a> · <a href="frame_selection.csv">Frame selection CSV</a></p>
<p class="note">These are paired Scripted Oracle/X-VLA qualitative illustrations, not empirical trained-TD3 failure comparisons or new success-rate trials. All Oracle failures are deliberately scripted, including opening the gripper during transport. X-VLA errors are from actual archived native neural rollouts, without introduced disturbances.</p>
<h2>Delivered scope</h2><p>Six paired cases in the main figure; twelve paired cases in four group pages; twelve detailed temporal pages; two policy-input/alignment strips; and two print-friendly main-figure halves. Each figure has PDF, editable-text SVG and 600 DPI PNG. The 432 unannotated PNG panels include full decoded workspace frames, common crops and native HD hand-eye frames.</p>
<h2>Checks and corrections</h2><ul><li>All 48 source streams were decoded fully. Simulation timestamps use saved physics ticks, including the 1/30 s presentation offset. Within-method camera frames are paired; event times need not match across methods.</li>
<li>Checksummed initial scenes match across each pair. Shared camera/crop never changes across its six selected moments.</li>
<li>Natural recovery Scene14 was replayed from an existing 14,500-update X-VLA benchmark trace. Baseline and instrumented actions, native observations/RPC payloads and physical states matched; no new VLA inference was needed. A no-fault Oracle counterpart was captured in the exact initial world.</li>
<li>The first added recovery render was rejected as blank. EGL had been loaded after scene reset. The v2 renderer loads it before visual shapes, rejects constant images, and matches lossless RGB/segmentation goldens from the original pinned physics binding.</li>
<li>Visual review caught header/time and policy-strip footer crowding; both were separated before delivery. Twelve unit tests also include blank-video rejection and hold/drop/reacquisition checks.</li>
<li>PDF fonts are embedded; SVG labels remain text. Source images remain genuine raster images, with unfilled vector highlights kept separate. Automated full-video decoding is not a claim that a human watched every source-video frame.</li></ul>
<h2>Case review</h2><p>Outcome and terminal XY distance columns are Oracle / X-VLA. Distances are diagnostic, not a replacement for hold/release/support/settle checks.</p>
<table><thead><tr><th>Case</th><th>Command</th><th>Outcomes</th><th>Terminal XY error</th><th>Visual interpretation / limits</th></tr></thead><tbody>'''
    report+=''.join(rows)+'</tbody></table>'
    report+='''<h2>Policy observation</h2><p>Scene13 shows actual RPC-verified front/wrist 224px inputs at 0, 3, 6, 12 and 18 s. A separate alignment figure links HD frames at 5.5 / 8.5 / 20.0 s to the latest preceding queries at 3 / 6 / 18 s, respectively. HD wrist frames are presentation cameras with different extrinsics, not actor inputs. The external view and privileged spatial inputs remained available, so disappearance from the wrist does not prove memory loss or semantic misidentification.</p>
<h2>Print and inference limits</h2><p>Use the two 180 mm-wide print halves for ordinary manuscript pages. The six-case main is tall; reducing it to fit a text-height page would also reduce its 8pt labels. Detailed pages are designed at 270 mm width. Source-camera resolution, not the 600 DPI export number, limits image detail.</p>
<p>Blue marks identify the target. Dashed marks indicate destinations, including schematic centres for textual regions, not success-tolerance boundaries. Completion uses the archived qualitative release/support/settle gate (4.5 cm XY threshold), not a strict 1 cm benchmark. All assistance, post-hoc selection and repeated-source caveats remain in caption.txt and evidence_limits.json.</p>
<p>Three ejection examples in the complete set illustrate different target shapes and spatial consequences, not three distinct model-intrinsic mechanisms. Only one is retained in the main figure. Scene14 adds a successful natural recovery; Scene02 remains as a straightforward success example. Scene12 reuses a previously published X-VLA source.</p></html>'''
    (out/'REPORT.html').write_text(report)
    page=(out/'index.html').read_text()
    if 'href="REPORT.html"' not in page:
        page=page.replace('</h1>','</h1><p><a href="REPORT.html">Read evidence / visual-review report</a></p>',1)
        (out/'index.html').write_text(page)
    if 'scene13_HD_query_alignment_preview.jpg' not in page:
        card='<section><h2>Scene13: HD events and latest actual policy query</h2><p>'
        card+='<a href="policy_observations/scene13_HD_query_alignment.pdf">PDF</a> · '
        card+='<a href="policy_observations/scene13_HD_query_alignment.svg">Editable SVG</a> · '
        card+='<a href="policy_observations/scene13_HD_query_alignment.png">600 DPI PNG</a></p>'
        card+='<img src="policy_observations/scene13_HD_query_alignment_preview.jpg"></section>'
        page=page.replace('<h2>Interpretation limits</h2>',card+'<h2>Interpretation limits</h2>')
        (out/'index.html').write_text(page)
    assets={}
    wanted=[]
    for figure in audit['figures']:wanted.extend(out/f for f in figure['files'])
    wanted += [out/f for f in ('index.html','REPORT.html','README.txt','REPRODUCE.txt','caption.txt','manifest.json',
        'frame_selection.csv','policy_alignment.json','unit_tests.txt','audit.json','status.json','delivery_checks.json','visual_review.json','evidence_limits.json')]
    for case in manifest['cases']:
        for method in case['methods']:
            wanted.append(out/method['dense_review_file'])
            for row in method['panels']:
                wanted.extend(out/item['file'] for item in row['files'].values())
    wanted.extend(out/item['file'] for item in manifest['policy_observations'])
    code=out/'reproduction_code';code.mkdir(exist_ok=True)
    import shutil
    for name in ('qualitative_figure.py','qualitative_recovery_source.py','qualitative_recovery_render.py',
                 'qualitative_delivery_audit.py','qualitative_policy_alignment.py','test_qualitative_figure.py'):
        dest=code/name;shutil.copy2(Path(__file__).parent/name,dest);wanted.append(dest)
    for path in sorted(set(wanted)):
        assets[str(path.relative_to(out))]={'sha256':digest(str(path)),'bytes':path.stat().st_size}
    write(out/'MANIFEST.sha256.json',{'scope':'final figure assets, unannotated panels, reports and reproduction source; does not hash itself',
        'assets':assets})
    print(json.dumps({'completed_delivery':True,'cases':len(manifest['cases']),
        'figures':len(figure_rows),'original_panels':panel_count,'checksummed_assets':len(assets)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args())
