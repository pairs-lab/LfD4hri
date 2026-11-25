"""Build the final, reviewable feedback Oracle delivery after all audits finish."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
import cv2
import imageio_ffmpeg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env_eval.audit_feedback import compare,score_run


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper'))
    args=parser.parse_args();root=args.root.resolve();project=root.parents[2]
    baseline=root.parent/'oracle_policy_style_v2_20261015'
    physics=json.loads((root/'results.json').read_text());presentation=json.loads((root/'presentation_audit.json').read_text())
    assert len(physics)==32 and all(r['overall_pass'] for r in physics)
    assert presentation['scenes_passed']==32 and presentation['videos_passed']==224
    protected=json.loads((root/'previous_videos_snapshot.json').read_text())
    unchanged=[{**r,'unchanged':digest(root.parent/r['file'])==r['sha256']} for r in protected]
    assert len(unchanged)==322 and all(r['unchanged'] for r in unchanged)
    (root/'previous_videos_preservation.json').write_text(json.dumps({'videos':len(unchanged),'unchanged':len(unchanged),'rows':unchanged},indent=2))
    comparisons=compare(root,baseline)
    assert len(comparisons)==32 and all(r['selection']['feasible'] for r in comparisons)
    replay=json.loads((root/'physical_replay_proof.json').read_text())
    assert len(replay)==32 and all(r['pass'] for r in replay)
    (root/'motion_comparison.json').write_text(json.dumps(comparisons,indent=2))
    validation=[]
    for p in sorted((root/'validation').glob('*/summary.json')):
        data=json.loads(p.read_text());validation.append({'name':p.parent.name,'episodes':data['episodes'],
            'successes':data['successes'],'strict_passes':data['feasible'],'file':str(p.relative_to(root))})
    (root/'validation_summary.json').write_text(json.dumps(validation,indent=2))
    summaries=[]
    for robot in ('UF850','UR5e_Robotiq2F'):
        rows=[r for r in comparisons if r['scene'].startswith(robot+'/')]
        summaries.append({'robot':robot,'episodes':len(rows),
            **{key:{'old_median':float(np.median([r['old'][key] for r in rows])),
                    'new_median':float(np.median([r['new'][key] for r in rows]))}
               for key in ('actual_joint_jerk_rms_rad_s3','filtered_joint_jerk_rms_rad_s3',
                           'joint_velocity_high_frequency_rms_rad_s','duration_seconds')}})
    (root/'quality_summary.json').write_text(json.dumps(summaries,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(metric,title) in zip(axes.flat,[('actual_joint_jerk_rms_rad_s3','Actual joint jerk RMS (rad/s³)'),
        ('filtered_joint_jerk_rms_rad_s3','Filtered joint jerk RMS (rad/s³)'),
        ('joint_velocity_high_frequency_rms_rad_s','High-frequency joint velocity RMS (rad/s)'),
        ('duration_seconds','Episode duration (s)')]):
        x=np.arange(2);a=[r[metric]['old_median'] for r in summaries];b=[r[metric]['new_median'] for r in summaries]
        ax.bar(x-.18,a,.36,label='Preserved v2',color='#9198a3');ax.bar(x+.18,b,.36,label='Feedback v3',color='#1976ae')
        ax.set_xticks(x,['UF850','UR5e']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'motion_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(robot,action,config) in zip(axes.flat,[('UF850','pick',1),('UF850','move',4),('UR5e_Robotiq2F','move',3),('UR5e_Robotiq2F','put',1)]):
        for base,label,color in ((baseline,'Preserved v2','#9198a3'),(root,'Feedback v3','#1976ae')):
            s=json.loads((base/robot/action/f'config{config}'/'trajectory.json').read_text())
            t=np.array([x['time'] for x in s]);tcp=np.array([x['tcp'] for x in s])
            ax.plot(t[1:],np.linalg.norm(np.diff(tcp,axis=0),axis=1)/np.diff(t),label=label,color=color,lw=1.3)
        ax.set(title=f'{robot} / {action} / C{config}',xlabel='Simulation time (s)',ylabel='Actual TCP speed (m/s)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'actual_tcp_speed.png',dpi=170);plt.close(fig)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();aux=[]
    for robot in ('UF850','UR5e_Robotiq2F'):
        parts=[root/robot/a/f'config{c}'/'view_1.mp4' for a,c in [('touch',2),('pick',1),('move',4),('put',1),('put',3)]]
        listing=root/f'{robot}_showcase_concat.txt';listing.write_text('\n'.join("file '"+str(p)+"'" for p in parts)+'\n')
        output=root/f'SHOWCASE_{robot}.mp4'
        if not output.exists():subprocess.run([ffmpeg,'-v','error','-n','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(output)],check=True)
        aux.append(output)
        output=root/f'COMPARE_{robot}_move4.mp4'
        if not output.exists():
            subprocess.run([ffmpeg,'-v','error','-n','-i',str(baseline/robot/'move/config4/view_1.mp4'),
                '-i',str(root/robot/'move/config4/view_1.mp4'),'-filter_complex',
                '[0:v]scale=960:540,setsar=1[l];[1:v]scale=960:540,setsar=1[r];[l][r]hstack=inputs=2:shortest=0[v]',
                '-map','[v]','-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(output)],check=True)
        aux.append(output)
    aux_audit=[]
    for p in aux:
        cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        for f in (0,n//2,n-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read();assert ok and frame.std()>5
        assert fps==30
        aux_audit.append({'file':p.name,'frames':n,'fps':fps,'duration_seconds':n/fps,'decode_initial_middle_final':True});cap.release()
    (root/'auxiliary_video_audit.json').write_text(json.dumps(aux_audit,indent=2))
    payload=[]
    for r in physics:
        robot='UF850' if r['robot']=='uf850' else 'UR5e_Robotiq2F'
        payload.append({'robot':robot,'action':r['action'],'config':r['config'],'prompt':r['task']['prompt'],
                        'path':f"{robot}/{r['action']}/config{r['config']}",'duration':r['duration_seconds']})
    html=(baseline/'index.html').read_text()
    html=re.sub(r'const rows=.*?;const film=',lambda _: 'const rows='+json.dumps(payload)+';const film=',html,flags=re.S)
    html=html.replace('Oracle · Native Robotiq','Oracle · Feedback motion')
    html=html.replace('Controller là privileged Oracle.','Controller là privileged Oracle có phản hồi trạng thái và lực.')
    (root/'index.html').write_text(html)
    # Source versions used by all rendering workers must agree with each other
    # and with the snapshot; packaging and validation-only tools are separate.
    run_versions=[json.loads(p.read_text()) for p in (root.parent/'v3_final_render_jobs').glob('*/run_sources.json')]
    assert len(run_versions)==4 and all(v==run_versions[0] for v in run_versions)
    assert all(digest(project/name)==h for name,h in run_versions[0].items())
    (root/'render_source_proof.json').write_text(json.dumps(run_versions[0],indent=2))
    for name in ('validate_feedback.py','package_feedback.py'):
        shutil.copy2(project/'env_eval'/name,root/'sources'/name)
    shutil.copy2(project/'env_eval/FEEDBACK_ORACLE.md',root/'sources/FEEDBACK_ORACLE.md')
    video_hashes={str(p.relative_to(root)):digest(p) for p in sorted(root.glob('*/*/*/*.mp4'))}
    assert len(video_hashes)==224
    manifest={'sources':{str(p.relative_to(root)):digest(p) for p in sorted((root/'sources').rglob('*')) if p.is_file()},
              'videos':video_hashes,'auxiliary_videos':{p.name:digest(p) for p in aux},
              'original_materials':json.loads((baseline/'manifest.json').read_text())['native_assets']}
    for name,h in manifest['original_materials'].items():assert digest(project/name)==h
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    reduction=[]
    for r in summaries:
        raw=r['actual_joint_jerk_rms_rad_s3'];hf=r['joint_velocity_high_frequency_rms_rad_s']
        reduction.append(f"| {r['robot']} | {raw['old_median']:.3f} → {raw['new_median']:.3f} | {(1-raw['new_median']/raw['old_median'])*100:.1f}% | {(1-hf['new_median']/hf['old_median'])*100:.1f}% |")
    active_comparison=json.loads((root/'qualified_motion_comparison.json').read_text())
    active_reductions=[]
    for robot in ('UF850','UR5e_Robotiq2F'):
        rows=[r for r in active_comparison if r['scene'].startswith(robot+'/')]
        a=float(np.median([r['old']['active_joint_jerk_rms_rad_s3'] for r in rows]))
        b=float(np.median([r['new']['active_joint_jerk_rms_rad_s3'] for r in rows]))
        active_reductions.append(f'{robot}: {(1-b/a)*100:.1f}%')
    normalized=json.loads((root/'tempo_normalized_audit.json').read_text())
    normalized_text='; '.join(f"{r['robot']}: giảm median {r['median_reduction_percent']:.1f}%, cải thiện {r['paired_cases_improved']}/16 ca"
                              for r in normalized['summary'])
    vtable='\n'.join(f"| {r['name']} | {r['episodes']} | {r['successes']} | {r['strict_passes']} |" for r in validation)
    force=max(r['max_touch_force_n'] for r in physics)
    mount=[]
    for robot in ('uf850','ur5e'):
        rows=[r for r in physics if r['robot']==robot]
        mount.append(f"- {robot}: lệch mount có tải lớn nhất {max(r['max_mount_translation_error_m'] for r in rows)*1000:.4f} mm; {max(r['max_mount_rotation_error_deg'] for r in rows):.4f}°.")
    profiles=json.loads((root/'selected_profiles.json').read_text())
    report=f'''# Oracle feedback v3 — audit chuyển động và tương tác, 2026-10-15

Đã hoàn thành **32/32 cấu hình danh định, 224/224 video đạt audit**. Mỗi robot có touch/pick/move/put × 4 config. Giữ nguyên **322 video** của hai bộ trước, đã đối chiếu SHA-256. Bộ này nằm trong thư mục mới.

## Xem kết quả

- [Bộ chọn đủ 32 cấu hình và 7 camera](index.html)
- [Showcase UF850](SHOWCASE_UF850.mp4), [Showcase UR5e](SHOWCASE_UR5e_Robotiq2F.mp4)
- [So sánh UF850](COMPARE_UF850_move4.mp4), [so sánh UR5e](COMPARE_UR5e_Robotiq2F_move4.mp4): trái là v2, phải là v3; cả hai chạy đúng thời gian mô phỏng, phía kết thúc trước giữ frame cuối. Không chèn chữ vào video.
- [Biểu đồ chất lượng chuyển động](motion_comparison.png), [vận tốc TCP thực tế](actual_tcp_speed.png)
- [Ảnh audit các góc](review/), [nhận xét thị giác](manual_review.json)

## Khảo sát và tiêu chí thiết kế

Đã đọc nguồn gốc và ghi phạm vi áp dụng trong [SURVEY.md](research/SURVEY.md): [ACT](https://arxiv.org/abs/2304.13705) dùng action chunks và temporal ensembling; [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) dùng chuỗi hành động với receding horizon; [CAPS](https://ai.bu.edu/caps/) chỉ ra rằng RL không tự bảo đảm độ mượt; [gentle manipulation](https://arxiv.org/abs/1903.08542) nhấn mạnh cân bằng thành công và lực tiếp xúc. [Ruckig](https://arxiv.org/abs/2105.04830) cung cấp giới hạn jerk cho bộ thực thi cổ điển.

Từ các nguồn này, tiêu chí áp dụng là: hành động liên tục và nhất quán, phản hồi theo trạng thái, phối hợp cổ tay–ngón kẹp, tiếp xúc nhẹ, không thêm nhiễu hoặc độ lắc để giả vẻ learning. Không có chuẩn thị giác chung nào chứng minh một trajectory đến từ RL.

## Thay đổi cụ thể

- Bộ thực thi giữ liên tục q, q̇, q̈ và giới hạn jerk tham chiếu; có preview trạng thái IK tương lai để giảm trễ. Giới hạn đã chọn: UF850 **{profiles['uf850']['joint_jerk']:.0f}**, UR5e **{profiles['ur5e']['joint_jerk']:.0f} rad/s³**. Đây là giới hạn lệnh, còn jerk robot thực tế được đo riêng bên dưới.
- Phân bổ lại thời gian quay cổ tay, giữ đường mang cong liên tục và bỏ các bend/lean ngẫu nhiên theo seed. Độ nghiêng nhỏ gắn với hướng vận chuyển, có bù vị trí tâm vật khi xoay.
- Ước lượng chuyển dịch vật và kết hợp dự đoán chồng lấn cho cùng thời điểm tương lai. Dự đoán dùng trạng thái mô phỏng, không có ACT/Diffusion network.
- Phản hồi lực trước gắp ngăn siết mạnh khi một ngón mới chạm vật. Đây là chế độ opt-in của class Robotiq cũ; API và mount của các caller cũ được giữ nguyên.
- Tính offset TCP trong frame của gripper để độ nhún của constraint không trở thành vòng phản hồi đẩy tay vào vật. Ràng buộc mặt bàn dùng collider pad và margin vật lý, hỗ trợ phối hợp khép ngón với hạ tay cho dao/thìa.
- Mở kẹp và nhấc nhẹ để thoát tiếp xúc được phối hợp; rút tay bắt đầu từ pose đo thực tế. Assist chỉ xuất hiện sau tiếp xúc chịu lực ở cả hai pad, không phải hút vật từ xa.
- Giữ native Robotiq, mount UF850 sát flange, mount UR5e riêng và palette UR5e gốc. Các asset gốc không bị sửa; manifest có hash kiểm chứng.

## So với v2 trên cùng 32 cấu hình

Các giá trị dưới đây là median của 16 episode mỗi robot, lấy từ khớp thực tế ở 30 Hz. Jerk RMS là sai phân bậc ba của q; high-frequency RMS dùng phần dư vận tốc sau cùng bộ lọc Savitzky–Golay 7 frame, bậc 3 cho cả hai phiên bản. Episode dài hơn có thể ảnh hưởng RMS; biểu đồ vận tốc và số liệu từng ca được cung cấp để kiểm tra tradeoff.

| Robot | Jerk RMS cũ → mới (rad/s³) | Giảm jerk | Giảm phần dư vận tốc tần số cao |
|---|---:|---:|---:|
{chr(10).join(reduction)}

Kiểm tra thêm khi loại các pha initial/final/settle khỏi RMS vẫn cho mức giảm **{'; '.join(active_reductions)}**. Vì vậy kết quả không chỉ do thêm frame đứng yên. Dữ liệu: [qualified_motion_comparison.json](qualified_motion_comparison.json); toàn bộ 32 video chạy lại đã đối chiếu đúng trạng thái với lượt qualification trong [physical_replay_proof.json](physical_replay_proof.json).

Để kiểm tra ảnh hưởng của chạy chậm, tính thêm chỉ số không thứ nguyên `RMS(jerk) × T³ / ‖q_cuối − q_đầu‖` trên các pha hoạt động. Chỉ số này bất biến khi chỉ kéo giãn đều thời gian. Kết quả: **{normalized_text}**. Đây là chỉ số kỹ thuật cho episode nhiều pha, không phải bộ phân loại learning/RL; không phải mọi metric ở mọi ca đều cải thiện. [Phương pháp và đủ 32 đối chiếu](tempo_normalized_audit.json).

Tệp [motion_comparison.json](motion_comparison.json) chứa đủ 32 đối chiếu, kể cả ca ít cải thiện hoặc chậm hơn. UF850 dành thêm thời gian cho cổ tay để tránh bão hòa giới hạn; không tăng tốc video để làm đẹp số liệu. Đây là lựa chọn chất lượng/nhịp chuyển động, không phải tối ưu thời gian ngắn nhất.

## Thành công và kiểm tra ngoài cấu hình danh định

Danh định: **UF850 16/16; UR5e 16/16**. Từng ca qua success/safety, chuyển động thực tế, camera, mount, đồng bộ inset và decode video. Các thử nghiệm dưới đây được báo riêng, không cộng video các góc thành số lượt thử độc lập.

| Bộ thử | Episode | Thành công tác vụ + safety | Đạt toàn bộ ngưỡng vật lý/chuyển động |
|---|---:|---:|---:|
{vtable}

Hai seed layout thay vị trí ban đầu của vật trong ±8 mm theo x/y. `small_force_pulse` dùng lực ngang bằng 1,5 lần trọng lượng vật trong 8 ms, giữa đoạn tiếp cận. Bộ `force_pulse` trước đó dùng cùng mức lực trong 1% thời gian tiếp cận (khoảng 33–55 ms); đây là stress test mạnh hơn, có thể làm vật trượt xa. Giữ nguyên cả những lần không đạt; xem [validation_summary.json](validation_summary.json) và từng `summary.json` để thấy nguyên nhân. Không suy rộng kết quả danh định thành bảo đảm mọi nhiễu đều thành công.

## Gắp, chạm, mount và camera

- Lực touch danh định lớn nhất **{force:.3f} N**, ngưỡng 5 N.
{chr(10).join(mount)}
- Audit mount kiểm tra cả có tải và 0,25 s cuối. Không dùng teleport hoặc gộp gripper vào robot để che lỗi mối ghép.
- Sáu góc cố định 1920×1080 và macro 1280×720, 30 FPS. Giữ các pose trước, trái, phải đã yêu cầu. Mỗi góc rộng có inset cận đồng bộ, không che vật chính theo segmentation.
- Cả hai pad phải nhìn thấy khi tương tác; vật phải nhìn thấy khi hạ, mở kẹp và sau rút tay. Các ca put kiểm tra vật không bị cắt khỏi khung cận.
- [presentation_audit.json](presentation_audit.json) ghi kiểm tra từng file encode, so pixel cùng frame của inset/macro, độ phân giải, thời lượng, chuyển động và mount. [render_source_proof.json](render_source_proof.json) chứng minh bốn job dùng cùng bản nguồn.

## Kết luận có giới hạn

Các số đo và ảnh trích hỗ trợ kết luận chuyển động êm hơn, ít dao động hơn và tương tác có kiểm soát hơn v2. Mức độ đẹp/chuyên nghiệp vẫn cần người xem đánh giá; ảnh audit là các mốc được trích, không phải tuyên bố đã xem thủ công mọi frame. Renderer và mesh vẫn có bề mặt đơn giản.

Đây là **privileged feedback Oracle**, có IK và assist sau tiếp xúc. Việc tối ưu profile bằng rollout reward là tìm kiếm tham số điều khiển, **không phải huấn luyện policy RL**. Kết quả nhằm cải thiện thực chất chất lượng chuyển động theo các nguyên tắc thường dùng trong robot learning; không khẳng định người xem có thể suy ra nguồn gốc learning từ video.

## Tái lập

Xem [hướng dẫn](sources/FEEDBACK_ORACLE.md), [profile đã chọn](selected_profiles.json), [36 lượt tuning](tuning_selection.json), [tests](tests.log), [manifest](manifest.json). Các video cũ được xác nhận nguyên byte trong [preservation](previous_videos_preservation.json).
'''
    (root/'REPORT.md').write_text(report)
    (root/'README.md').write_text('# Feedback Oracle v3\n\n[Video selector](index.html) · [Audit report](REPORT.md) · [Survey](research/SURVEY.md)\n\n32 scenes, 224 synchronized camera clips, two showcases and two comparisons. Original robot/gripper appearance preserved. The controller is a privileged model-based Oracle.\n')
    print('Packaged: 32 scenes, 224 clips, four auxiliary films; 322 previous films unchanged')


if __name__=='__main__':main()
