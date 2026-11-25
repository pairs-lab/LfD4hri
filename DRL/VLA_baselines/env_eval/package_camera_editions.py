"""Package both camera extensions after complete, independent video audits."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
import cv2,imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
from env_eval.camera_editions import SETTINGS,ENGINE_FILES,dump
from env_eval.audit_camera_editions import decode_at


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def showcase(root,label):
    parts=[root/label/action/f'config{config}'/'view_1.mp4' for action,config in [('touch',2),('pick',1),('move',4),('put',1),('put',3)]]
    listing=root/f'showcase_{label}.txt';listing.write_text(''.join("file '"+str(p.resolve())+"'\n" for p in parts))
    target=root/f'SHOWCASE_{label}.mp4'
    if not target.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-n',
        '-f','concat','-safe','0','-i',str(listing),'-an','-c','copy','-movflags','+faststart',str(target)],check=True)
    check=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',str(target),'-f','null','-'],capture_output=True,text=True)
    cap=cv2.VideoCapture(str(target),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    expected=0
    for path in parts:
        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);expected+=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    assert check.returncode==0 and not check.stderr.strip() and count==expected and fps==30
    return dict(file=target.name,frames=count,fps=fps,full_decode_pass=True,sha256=digest(target))


def event_indices(samples,action):
    approach=[i for i,s in enumerate(samples) if s['phase']=='approach']
    held=[i for i,s in enumerate(samples) if s['held']]
    touch=[i for i,s in enumerate(samples) if s['physical_contact']]
    carry=[i for i,s in enumerate(samples) if s['phase']=='carry']
    release=[i for i,s in enumerate(samples) if s['phase']=='release']
    return [approach[len(approach)//2] if approach else 0,
        held[0] if held else touch[0] if touch else len(samples)//2,
        release[len(release)//2] if release else carry[len(carry)//2] if carry else touch[-1] if touch else len(samples)//2,
        len(samples)-1]


def sheets(root):
    out=root/'review';out.mkdir(exist_ok=True)
    for label in ('UF850','UR5e_Robotiq2F'):
        for action in ('touch','pick','move','put'):
            canvas=Image.new('RGB',(1920,1248),'#111923');draw=ImageDraw.Draw(canvas)
            native=Image.new('RGB',(1920,1248),'#111923');nd=ImageDraw.Draw(native)
            for config in range(1,5):
                directory=root/label/action/f'config{config}';samples=json.loads((directory/'trajectory.json').read_text())
                for column,index in enumerate(event_indices(samples,action)):
                    x=480*column;y=(config-1)*312
                    title=f'{label} / {action} / config{config} / {samples[index]["phase"]} / {index/30:.2f}s'
                    draw.text((x+8,y+7),title,fill='white');nd.text((x+8,y+7),title,fill='white')
                    frame=decode_at(directory/'view_1.mp4',index)
                    eye=decode_at(directory/'handeye.mp4',index)
                    canvas.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((480,270)),(x,y+31))
                    native.paste(Image.fromarray(cv2.cvtColor(eye,cv2.COLOR_BGR2RGB)).resize((480,270)),(x,y+31))
            canvas.save(out/f'{label}_{action}_layout.jpg',quality=93)
            native.save(out/f'{label}_{action}_handeye.jpg',quality=93)


def html(root,mode,rows,base_name):
    naming=json.loads(SETTINGS.read_text())['folder_suffixes']
    editions={mode:base_name+suffix for mode,suffix in naming.items()}
    text='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Oracle · Three camera editions</title>
<style>*{box-sizing:border-box}body{margin:0;background:#10151d;color:#ebf0f7;font:15px system-ui}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 7px}p,small{color:#a9b6c8}a{color:#96c7ff}.toolbar,#views{display:flex;gap:9px;flex-wrap:wrap;margin:16px 0}select,button{background:#1c2838;color:inherit;border:1px solid #40516a;border-radius:7px;padding:10px;font:inherit}button.active{background:#205e9b}video{display:block;width:100%;max-height:78vh;aspect-ratio:16/9;background:#080c12;border-radius:8px;object-fit:contain}.links{display:flex;gap:16px;flex-wrap:wrap}.meta{margin:10px 0 20px;color:#bdc9d9}</style>
<main><h1>Oracle · Ba bố cục camera</h1><p>32 cấu hình · UF850 & UR5e · 6 góc rộng · hai khung cận cùng kích thước · 30 FPS</p>
<div class="links"><a href="REPORT.md">Báo cáo audit</a><a href="camera_settings.json">Camera settings</a><a id="showcaseUF">Showcase UF850</a><a id="showcaseUR">Showcase UR5e</a></div>
<div class="toolbar"><select id="edition"><option value="front">Front gripper · v3 gốc</option><option value="front_handeye">Front trên + hand-eye dưới</option><option value="handeye">Hand-eye thay front</option></select><select id="robot"><option>UF850</option><option>UR5e_Robotiq2F</option></select><select id="action"><option>touch</option><option>pick</option><option>move</option><option>put</option></select><select id="config"><option value="1">Config 1</option><option value="2">Config 2</option><option value="3">Config 3</option><option value="4">Config 4</option></select></div>
<video id="film" controls playsinline preload="metadata"></video><div id="views"></div><div class="meta" id="meta"></div><small>Video không chèn chữ. Chuyển bố cục giữ cùng thời điểm để so sánh. Đây là feedback Oracle có IK và hỗ trợ giữ vật sau tiếp xúc hai pad.</small></main>
<script>
const rows=__ROWS__,editions=__EDITIONS__;const el=id=>document.getElementById(id);let view='view_1';el('edition').value=__MODE__;
const files=['view_1','view_2','view_3','view_4','view_5','view_6','gripper_closeup','handeye'];const names=['Góc rộng 1','Góc rộng 2','Workspace','Trước','Trái','Phải','Cận front gripper','Hand-eye'];
function updateControls(){[['robot','robot'],['action','action'],['config','config']].forEach(([id,key])=>{const previous=el(id).value;const candidates=rows.filter(r=>id==='robot'||r.robot===el('robot').value).filter(r=>id!=='config'||r.action===el('action').value);const values=[...new Set(candidates.map(r=>String(r[key])))];el(id).replaceChildren(...values.map(value=>new Option(id==='config'?'Config '+value:value,value)));if(values.includes(previous))el(id).value=previous});el('showcaseUF').hidden=rows.filter(r=>r.robot==='UF850').length!==16;el('showcaseUR').hidden=rows.filter(r=>r.robot==='UR5e_Robotiq2F').length!==16}
function load(keep=false){const edition=el('edition').value;const r=rows.find(r=>r.robot===el('robot').value&&r.action===el('action').value&&r.config===Number(el('config').value));const allowed=f=>!(f==='handeye'&&edition==='front')&&!(f==='gripper_closeup'&&edition==='handeye');if(!allowed(view))view='view_1';const t=keep?el('film').currentTime:0;const play=keep&&!el('film').paused;const base='../'+editions[edition]+'/';el('film').src=base+r.path+'/'+view+'.mp4';el('film').onloadedmetadata=()=>{el('film').currentTime=Math.min(t,el('film').duration);if(play)el('film').play()};el('meta').textContent=r.prompt+' · '+r.duration.toFixed(2)+' s';el('showcaseUF').href=base+'SHOWCASE_UF850.mp4';el('showcaseUR').href=base+'SHOWCASE_UR5e_Robotiq2F.mp4';el('views').replaceChildren();files.forEach((f,i)=>{if(!allowed(f))return;const b=document.createElement('button');b.textContent=names[i];b.classList.toggle('active',f===view);b.onclick=()=>{view=f;load(true)};el('views').append(b)})}
['robot','action','config'].forEach(id=>el(id).onchange=()=>{updateControls();load()});el('edition').onchange=()=>load(true);updateControls();load();
</script></html>'''
    text=text.replace('__ROWS__',json.dumps(rows)).replace('__EDITIONS__',json.dumps(editions)).replace('__MODE__',json.dumps(mode))
    (root/'index.html').write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('env_eval/demos/oracle_feedback_v3_equal_camera_sources_20261015'))
    p.add_argument('--baseline',type=Path,default=Path('env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper'));a=p.parse_args()
    baseline=a.baseline.resolve();cache=a.cache.resolve();project=Path(__file__).resolve().parents[1]
    settings=json.loads(SETTINGS.read_text());metadata=json.loads((cache/'run_sources.json').read_text())
    assert settings==metadata['camera_settings']
    assert all(digest(project/f)==h for f,h in metadata['controller_sources'].items())
    protected=json.loads((baseline/'previous_videos_snapshot.json').read_text())
    original=json.loads((baseline/'manifest.json').read_text())
    for key in ('videos','auxiliary_videos'):
        protected += [dict(file=str(Path(baseline.name)/file),sha256=h) for file,h in original[key].items()]
    preservation=[dict(**row,unchanged=digest(baseline.parent/row['file'])==row['sha256']) for row in protected]
    assert len(preservation)==550 and all(r['unchanged'] for r in preservation)
    assert all(digest(project/file)==h for file,h in original['original_materials'].items())
    proof=[];rows=[]
    for path in sorted(cache.glob('*/*/*/audit.json')):
        relative=path.parent.relative_to(cache);samples=json.loads(path.with_name('trajectory.json').read_text())
        old=json.loads((baseline/relative/'trajectory.json').read_text());audit=json.loads(path.read_text())
        with np.load(path.with_name('joint_reference.npz')) as new_ref,np.load(baseline/relative/'joint_reference.npz') as old_ref:
            reference_exact=bool(set(new_ref.files)==set(old_ref.files) and all(np.array_equal(new_ref[key],old_ref[key]) for key in old_ref.files))
        proof.append(dict(scene=str(relative),pass_exact=bool(old==samples and reference_exact),frames=len(samples),
            fields_compared=list(old[0]),physics_reference_240hz_exact=reference_exact))
        rows.append(dict(robot=relative.parts[0],action=audit['action'],config=audit['config'],path=str(relative),prompt=audit['task']['prompt'],duration=audit['duration_seconds']))
    assert len(proof)==32 and all(r['pass_exact'] for r in proof)
    base_name=baseline.name.removesuffix(settings['folder_suffixes']['front'])
    for mode in ('front_handeye','handeye'):
        root=baseline.with_name(base_name+settings['folder_suffixes'][mode])
        audits=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/*/camera_audit.json'))]
        assert len(audits)==32 and all(r['pass'] for r in audits)
        films=[v for r in audits for v in r['videos']];expected=256 if mode=='front_handeye' else 224
        assert len(films)==expected and all(v['pass_video'] for v in films)
        # The user's latest size constraint applies to every view/config.
        dimensions=[]
        for path in root.glob('*/*/*/layout.json'):
            for view,layout in json.loads(path.read_text()).items():
                if mode=='front_handeye':assert layout['rects'][0][3:]==layout['rects'][1][3:]
                dimensions.append(dict(scene=str(path.parent.relative_to(root)),view=view,rects=layout['rects']))
        panel_sizes=sorted({tuple(rect[3:]) for d in dimensions for rect in d['rects']})
        size_text=', '.join(f'{w}×{h} px' for w,h in reversed(panel_sizes))
        dump(root/'camera_settings.json',settings);dump(root/'layouts.json',dimensions)
        dump(root/'physical_replay_proof.json',proof)
        dump(root/'previous_videos_preservation.json',dict(videos=550,unchanged=550,rows=preservation))
        dump(root/'presentation_audit.json',dict(scenes=32,scenes_passed=32,videos=expected,videos_passed=expected,rows=audits))
        edition_results=[]
        for r in rows:
            directory=root/r['path'];physical=json.loads((cache/r['path']/'audit.json').read_text())
            physical.update(camera_edition=mode,videos=sorted(p.name for p in directory.glob('*.mp4')))
            temporary=directory/'audit.edition.json';dump(temporary,physical);temporary.replace(directory/'audit.json')
            edition_results.append(physical)
        dump(root/'results.json',edition_results)
        dump(root/'wide_camera_settings.json',edition_results[0]['camera_views'])
        dump(root/'source_proof.json',metadata)
        presence=cache/'all_frame_camera_presence.json'
        if presence.exists():shutil.copy2(presence,root/presence.name)
        smoke=baseline.parent/'camera_layout_trials_20261015/all_three_smoke_proof.json'
        if smoke.exists():shutil.copy2(smoke,root/'all_three_smoke_proof.json')
        extras=[showcase(root,label) for label in ('UF850','UR5e_Robotiq2F')]
        dump(root/'showcase_audit.json',extras);sheets(root);html(root,mode,rows,base_name)
        sources=root/'sources';sources.mkdir(exist_ok=True)
        for file in ['camera_editions.py','camera_editions.json','camera_gallery.py','handeye_camera.py','audit_camera_editions.py','finish_camera_editions.py','patch_wide_camera.py','package_camera_editions.py','CAMERA_EDITIONS.md']:
            shutil.copy2(project/'env_eval'/file,sources/file)
        for file in ENGINE_FILES:shutil.copy2(project/file,sources/Path(file).name)
        shutil.copy2(project/'env_eval/tests/test_camera_editions.py',sources/'test_camera_editions.py')
        shutil.copy2(cache/'logs/tests.log',root/'tests.log')
        minima={name:dict(object_pixels=min(r['cameras'][name]['minimum_critical_object_pixels'] for r in audits),
            pad_pixels=[min(r['cameras'][name]['minimum_interaction_pad_pixels'][i] for r in audits) for i in (0,1)],
            border_px=min(r['cameras'][name]['minimum_critical_object_border_px'] for r in audits)) for name in audits[0]['cameras']}
        dump(root/'quality_summary.json',dict(camera_minima=minima,equal_panel_dimensions=True,panel_sizes_px=panel_sizes,unchanged_previous_films=550))
        title='Front gripper trên + hand-eye dưới' if mode=='front_handeye' else 'Hand-eye thay front gripper'
        report=f'''# Oracle v3 — {title}

Hoàn thành **32/32 cấu hình**, **{expected}/{expected} clip từng cấu hình đạt audit**, cộng hai showcase. UF850 16/16 và UR5e 16/16, mỗi robot touch/pick/move/put × 4 config. [Mở bộ chọn cả ba bố cục](index.html).

## Bố cục đã chốt

Sáu góc rộng 1920×1080, 30 FPS. Front và hand-eye đều 1280×720. Kích thước khung phụ được dùng: **{size_text}**. Mặc định bằng front gốc 594×334; khi góc cũ sẽ che vật/đích, render lại góc rộng sạch và chọn góc/vị trí nhỏ hơn. Ở bản hai khung, hai khung **luôn cùng kích thước**, front ở trên, hand-eye bên dưới, cách 12 px. Không kéo méo, không crop mất vùng thao tác, không chèn chữ. Vị trí khung phụ cố định suốt từng clip. [Tọa độ đầy đủ](layouts.json).

Hand-eye gắn cứng theo frame COM của tool như VLA. Đã hiệu chỉnh extrinsics cho Robotiq native theo ảnh tham chiếu của người dùng: eye `(0,.065,.05)`, target `(0,0,.36)`, up `(0,1,0)`, FOV dọc 90°. Mở rộng FOV ngang để khung 16:9 có cùng kích thước front, vẫn thấy hai ngón phía dưới và vùng thao tác ở trước. [Setting lưu sẵn](camera_settings.json), [cách chạy ra ba thư mục](sources/CAMERA_EDITIONS.md).

## Bằng chứng audit

- Decode **toàn bộ frame** của từng video; kiểm tra FPS, độ phân giải và số frame bằng camera nguồn.
- Vật chính và hai pad phải nhìn thấy trong tất cả frame tương tác, vật tiếp tục nhìn thấy khi mở kẹp/rút tay/settle ở tác vụ put. Ngưỡng native: vật ≥100 px, mỗi pad ≥20 px, lề vật ≥8 px. [Số nhỏ nhất đo được](quality_summary.json).
- Thêm kiểm tra mọi frame từ đầu đến cuối: gripper và vật luôn xuất hiện trong hand-eye, kể cả ngoài pha tương tác. [Every-frame camera presence](all_frame_camera_presence.json).
- Kiểm tra footprint khung mới với AABB toàn vật/đích trong từng frame; inset gốc dùng segmentation đã đạt ở v3. Không khung mới nào che vật chính/đích. Mọi inset so với camera native ở nhiều mốc trong cùng clip phải có sai số pixel trung bình <6. [Audit từng clip](presentation_audit.json).
- **32/32 replay giống hệt toàn bộ telemetry từng frame** của v3, gồm khớp, TCP, vật, finger joints, trạng thái gắp, lực, mount; các mảng lệnh q/q̇/q̈ ở **240 Hz cũng giống hệt**. Chỉ đổi cách quay/ghép. [Proof](physical_replay_proof.json).
- Giữ nguyên byte **550/550 video đã hoàn thành trước đó** (v1, v2 và v3); material asset và controller không thay đổi. [Đối chiếu SHA-256](previous_videos_preservation.json), [nguồn chạy](source_proof.json), [manifest](manifest.json).
- Ảnh audit bao phủ cả 32 cấu hình, các mốc tiếp cận/tiếp xúc/mang hoặc thả/kết thúc, ở hình ghép và hand-eye native: [review](review/). Nhận xét xem mẫu ảnh được lưu riêng trong `manual_review.json`; không coi ảnh mẫu là đã xem tay mọi frame.

## Video và phạm vi kết luận

[Showcase UF850](SHOWCASE_UF850.mp4) · [Showcase UR5e](SHOWCASE_UR5e_Robotiq2F.mp4). Sáu góc vẫn gồm front/trái/phải theo pose đã yêu cầu. Bản gốc [front gripper](../{baseline.name}/index.html) và báo cáo chuyển động [v3](../{baseline.name}/REPORT.md) được giữ nguyên.

Đây là privileged feedback Oracle có IK và assist sau tiếp xúc chịu lực hai pad, không phải policy RL/VLA đã huấn luyện. Success 100% ở **32 cấu hình danh định** không phải bảo đảm ngoài phân phối; các ca nhiễu không đạt vẫn được ghi trong báo cáo v3. Các bộ camera mới dùng đúng cùng trajectory nên giữ các chỉ số chuyển động và giới hạn đó.

## Tái lập

[Settings](camera_settings.json) · [hướng dẫn ba mode](sources/CAMERA_EDITIONS.md) · [thử xuất đủ ba mode](all_three_smoke_proof.json) · [tests](tests.log). Bộ render nguồn được giữ ở `{cache.name}`; các bản thử khung vuông lớn cũng được giữ trong `camera_layout_trials_20261015`, không xóa video thử.
'''
        (root/'REPORT.md').write_text(report)
        files=list(root.glob('*/*/*/*.mp4'))+[root/e['file'] for e in extras]
        manifest=dict(videos={str(p.relative_to(root)):digest(p) for p in files},
            sources={str(p.relative_to(root)):digest(p) for p in sources.iterdir() if p.is_file()},original_materials=original['original_materials'])
        dump(root/'manifest.json',manifest)
        print('PACKAGED',root.name,'scenes',32,'videos',expected,'showcases',2,'protected',550,flush=True)


if __name__=='__main__':main()
