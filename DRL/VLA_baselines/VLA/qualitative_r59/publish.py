"""Publish findings recorded AFTER actual view_image inspection in R59."""
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .review import SOURCE, OUT, METHODS, read

# These are manual visual findings, not labels guessed from fault names.
FINDINGS = {
 'scene01': ('Khối xanh được nâng, mang tới và thả nằm trong bát tím; tay Oracle rút ra.',
             'Khối xanh nằm trong bát tím sau mở kẹp; không còn nằm giữa hai pad.'),
 'scene02': ('Khối đỏ được đặt và giữ ổn định trên khối cam sau khi Oracle rút tay.',
             'Khối đỏ được thả lên khối cam và giữ ổn định; tay không mang khối đỏ đi tiếp.'),
 'scene03': ('Khối xanh đang được giữ trước 4.47s, tách khỏi ngón trong 4.53–4.87s, rơi ngoài bát xanh; tay tiếp tục đi rỗng.',
             'Khối xanh được mang đúng bát xanh, thả vào bát và đứng yên ở cuối.'),
 'scene04': ('Hai ngón đã đóng khi hạ tới khối hồng; khối nghiêng bên ngón nhưng vẫn ở nguồn ở 5.60s. Gắp thất bại, không phải cú đâm mạnh.',
             'Khối hồng được nâng và thả vào bát xanh; vật còn trong bát ở 8.20s.'),
 'scene05': ('Đã giữ khối hồng nhưng còn ở phía nguồn, chưa tới bát vàng ở 22.0s. Hình ảnh chứng minh vận chuyển chưa xong; log chứng minh timeout có chủ ý do giới hạn tốc độ.',
             'Khối hồng được mang vào bát vàng và thả ổn định trước 8.70s.'),
 'scene06': ('Khối hồng được đặt lên khối đỏ rồi Oracle rút tay; vị trí đích nhìn rõ.',
             '8.93–9.43s: khối hồng từ cạnh ngón bật/văng qua mép vùng đen ra nền trắng; còn ngoài vùng thao tác ở 20s. Không có hold xác nhận trước va chạm, nên không gọi là rơi khi vận chuyển.'),
 'scene07': ('Khối xanh được đặt trên khối tím và vẫn đứng vững sau rút tay.',
             '9.00–9.40s: khối xanh trượt/văng ngang ra mép vùng đen, tách xa kẹp; tay tiếp tục di chuyển rỗng và không đưa vật lên khối tím trước 20s. Không gọi đây là nảy cao theo trục Z.'),
 'scene08': ('Quả peach được nâng và đặt ổn định trên khối cam; Oracle rút tay.',
             '14.93–15.33s: peach tách khỏi hai ngón và văng ra phía trước, nằm xa khối cam tới 20s. Log không có verified hold, nên không gán nhãn transit drop.'),
 'scene09': ('Hộp gelatin đỏ đang được mang, sau mở kẹp rơi xuống ngoài khối vàng khoảng 4.57–4.90s; tay tiếp tục rỗng.',
             '12.00–12.40s: hộp đỏ bị đẩy/văng sang trái, rồi ở lại xa khối vàng; kẹp nâng lên rỗng. Bát cyan là distractor, KHÔNG phải đích của lệnh này.'),
 'scene10': ('Kẹp đóng hạ sát quả cam nhưng không mang vật lên; vật vẫn ở nguồn tới kết thúc 5.60s. Không có bằng chứng ground slam dữ dội.',
             '8.63–9.43s: ngón khép cạnh quả cam rồi nâng rỗng; vật ở nguyên chỗ. Lặp tiếp cận vẫn không đem vật tới góc dưới-phải trước 20s. Có tiếp xúc bàn trong log, chưa đủ quy nguyên nhân cho model.'),
 'scene11': ('Peach đã ở giữa ngón trong khi vận chuyển, rơi khoảng 4.70–5.03s xa vùng bên-phải; tay đi tiếp rỗng.',
             '9.53–10.33s: kẹp nâng cao rỗng còn peach ở trên bàn; 14.77s lại tiếp cận nhưng vẫn không mang tới đích trước 20s. Audit camera policy riêng cho thấy front luôn còn thấy vật ở cả7query, wrist mất vật tại query60/120/180; GT vẫn được cấp. Không kết luận quên vật từ camera quay.'),
 'backup01': ('Strawberry được mang tới góc dưới-phải và đặt xuống; tay rút.',
              'Strawberry được mang tới góc dưới-phải và thả xuống ổn định.'),
 'backup02': ('Quả trắng còn được giữ ở phía nguồn khi hết 22s; chưa tới khối xám. Giới hạn tốc độ có script được công khai.',
              'Quả trắng được đặt trên khối xám và vẫn ở đó sau mở kẹp.'),
 'backup03': ('Strawberry tách khỏi tay đang vận chuyển và rơi ngoài khối xanh; ngón tiếp tục rỗng.',
              'Strawberry được mang và đặt lên khối xanh, không phải một clip gắp hụt được gán success.'),
 'backup04': ('Oracle vẫn tìm được, mang và đặt quả trắng lên khối hồng dù script có offset. Đây là SUCCESS phản chứng, không gán fail theo tên persistent_miss.',
              'Quả trắng được mang và đặt ổn định trên khối hồng.')
}


def main():
    inventory = read(OUT/'inventory.json')
    integrity = read(OUT/'independent_delivery_integrity.json')
    if not integrity['passed']: raise ValueError('Immutable source integrity failed')
    for name in ('FINAL_AUDIT.json', 'audit/reference_camera_extrinsics.json', 'audit/whole_stream_packet_PTS.json'):
        if not read(SOURCE/name)['passed']: raise ValueError('Source media/camera gate failed: '+name)
    rows = []; html = ['<!doctype html><meta charset="utf-8"><title>R59 independent frame review</title>',
        '<style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:0 18px}img{width:100%}video{width:100%}section{border-top:1px solid #aaa;margin:28px 0}details{margin:12px 0}p{line-height:1.5}</style>',
        '<h1>R59: independent encoded-frame review</h1>',
        '<p>22 video chính / 11 cặp + 8 backup / 4 cặp. Tất cả PUT, UR5e + Robotiq2F-85. '
        'Đây là selected qualitative examples, không phải một mẫu để tính success rate.</p>',
        '<p><b>RL là tên thư mục cho scripted privileged Oracle, không phải RL policy đã huấn luyện.</b> '
        'X-VLA nhận simulator GT; video là replay trung thực của action neural đã lưu, có mở kẹp cuối được công khai. '
        'Không dùng các lỗi được script của Oracle để suy ra nhược điểm tự nhiên của RL.</p>',
        '<p>Ảnh dưới là frame thật decode lại từ video; event zoom dùng crop cố định chỉ để review. '
        'Video gốc không crop/cắt/tăng tốc/thêm chữ. Các mốc hiển thị là simulation time; encoded PTS bắt đầu sớm hơn 1/30 giây.</p>']
    html.append('<p>Hand-eye trình diễn không phải camera wrist của policy. Kết luận về policy visibility phải lấy từ audit observation riêng, không từ ảnh quay đẹp.</p>')
    lines = ['# R59 — kết quả tự review độc lập', '',
        'Kết luận: **đủ 22 video chính đúng nhóm 2/3/3/3, cùng 8 video backup**. '
        'Thất bại tác vụ trong 12 clip failure chính là rõ; không cần quay lại chỉ để làm lỗi trông dữ dội hơn.', '',
        'Đã thực sự mở 30 contact sheet event zoom (12 mốc/rollout), 12 chuỗi hand-eye của toàn bộ main failure, '
        'và 2 overview không crop để phản chứng. Đây là review mẫu frame dày quanh sự kiện, không tuyên bố xem thủ công mọi frame liên tục.', '',
        '## Phân biệt outcome với nguyên nhân', '',
        '- VLA: scene 6–9 va chạm làm vật tách/văng; scene 10–11 gắp hụt, nâng và vận chuyển kẹp rỗng, chưa phục hồi được trước timeout.',
        '- Oracle: scene 3/9/11 rơi khi đang vận chuyển; scene 4/10 tiếp cận bằng kẹp đóng nên gắp thất bại; scene 5 vận chuyển chậm có chủ ý và hết ngân sách 22s.',
        '- **Không đủ chứng cứ** cho misidentification, mất trí nhớ, VLA transit drop hoặc violent ground/table slam. Không gán các nhãn đó cho video hiện có.',
        '- Scene 10/11 VLA có tiếp xúc bàn thật; grasp-height/controller/safety/data vẫn là giả thuyết cạnh tranh. Video không chứng minh lỗi hoàn toàn của model.',
        '- Backup04 Oracle thực sự thành công dù có offset: giữ nguyên phản chứng này.', '',
        '## Từng cặp và điều nhìn thấy', '']
    for record in inventory['records']:
        label, method = record['scene'], record['method']
        note = FINDINGS[label][METHODS.index(method)]
        opened = [method+'_event_zoom.jpg']
        if label.startswith('scene') and not record['task_success']:
            opened.append(method+'_handeye.jpg')
        if (label, method) in (('scene05', METHODS[0]), ('scene06', METHODS[1])):
            opened.append(method+'_uncropped_overview.jpg')
        hashes = {str(Path(label)/n): sha256_file(OUT/label/n) for n in opened}
        row = {'scene': label, 'method': method, 'actual_visual_note': note,
            'task_success': record['task_success'], 'visually_clear_task_outcome': True,
            'all_original_requested_failure_mechanisms_demonstrated': False,
            'review_scope': 'actual sampled encoded frames; not continuous every-frame playback',
            'actually_opened_review_image_sha256': hashes, 'source_frames': record['source_frame_indices']}
        rows.append(row)
        atomic_json(OUT/label/(method+'_MANUAL_FINDINGS.json'), row)
        lines.append(f"- **{label} / {method}**: {note}")
        src = Path(record['source_pair'])/method/'view_2.mp4'
        link = os.path.relpath(src, OUT)
        status = 'SUCCESS' if record['task_success'] else 'FAILURE'
        html += [f'<section><h2>{label} / {method} — {status}</h2><p>{escape(note)}</p>',
            f'<video controls preload="none" src="{escape(link)}"></video>',
            f'<p><a href="{escape(link)}">Video gốc đầy đủ</a> · '
            f'<a href="{escape(os.path.relpath(Path(record["source_pair"])/"index.html", OUT))}">Cả sáu góc + hand-eye + log</a></p>',
            f'<details><summary>12 frame event zoom — crop cố định, chỉ dành cho review</summary><img loading="lazy" src="{label}/{method}_event_zoom.jpg"></details>',
            f'<details><summary>12 frame native hand-eye</summary><img loading="lazy" src="{label}/{method}_handeye.jpg"></details>',
            f'<details><summary>Overview không crop</summary><img loading="lazy" src="{label}/{method}_uncropped_overview.jpg"></details></section>']
    lines += ['', '## Camera và phạm vi giao', '',
        '210 stream gốc vẫn nguyên hash. Tái xác minh 2.207 entry trong manifest; '
        'kế thừa kết quả full decode/PTS 30FPS và 44 bộ probe extrinsics đã xác minh, vì toàn bộ video và audit không thay đổi. '
        'Mỗi rollout có 6 góc 1920×1080 và native hand-eye 1280×720, H.264/yuv420p/30FPS; inset cố định theo cặp, '
        'EGL, ánh sáng và phép chuyển COM giữ nguyên.', '',
        '**Bộ này là một robot UR5e, mode hand-eye-only. Không phải bộ mới đủ cả UF850 và cả ba mode front/front+handeye/handeye.** '
        'Các góc wide và timing của mode hiện tại khớp chuẩn tham chiếu; không nói đã hoàn thành mọi phạm vi của prompt quay cũ.', '',
        '**Camera trình diễn khác camera policy.** Scene11: camera front policy thấy target tại đủ7query; wrist policy mất target ở query60/120/180. '
        'Presentation hand-eye có60frame mất vật, nhưng wide vẫn cho thấy vật ở bàn. Không dùng ảnh presentation để suy ra model mất hết thông tin. '
        'Số liệu gốc: `../ur5e_xvla_put_r58_complete_20261017/audit/camera_visibility_limits.json`.', '',
        '## Vì sao không cần dựng/quay lại failure', '',
        'Outcome đã nhìn thấy: vật ngoài nguồn đích, kẹp rỗng, hoặc vật chưa được mang tới đích khi hết ngân sách. '
        'Làm va chạm mạnh thêm bằng thay action VLA sẽ không còn là failure thật của checkpoint. '
        'Các cảnh Oracle cố ý gây lỗi phải luôn giữ disclosure. Clip này hỗ trợ minh họa hành vi quan sát được, '
        'không đủ để khẳng định VLA yếu hơn learned RL nói chung.', '',
        '[Gallery frame và video](index.html). [Bộ video gốc](../ur5e_xvla_put_r58_complete_20261017/index.html).', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'index.html').write_text('\n'.join(html))
    atomic_json(OUT/'REVIEW_COMPLETE.json', {'passed': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'main_pairs': 11, 'main_videos': 22, 'backup_pairs': 4, 'backup_videos': 8,
        'manually_opened_images': sum(len(r['actually_opened_review_image_sha256']) for r in rows),
        'actual_frame_review_complete_for_all_30_rollouts': True, 'same_immutable_delivery': integrity,
        'rerun_required_for_current_truthful_outcome_labels': False,
        'original_all_mechanism_and_all_robot_camera_mode_requests_NOT_all_satisfied': True,
        'all_failures_due_to_model_alone_NOT_established': True, 'findings': rows})
    print({'passed': True, 'rollouts': len(rows), 'manual_images': sum(len(r['actually_opened_review_image_sha256']) for r in rows)})


if __name__ == '__main__': main()
