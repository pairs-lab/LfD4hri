"""Read-only artifact reporter; never selects frames or changes rollout labels."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import time

from .common import OUT,read
from VLA.common.artifact_io import atomic_json


def optional(path,fallback):
    try:return read(path)
    except (FileNotFoundError,json.JSONDecodeError):return fallback


def publish():
    ledger=optional(OUT/'candidate_ledger.json',[])
    owners=[optional(p,{}) for p in sorted(OUT.glob('automatic_v*/status.json'))]
    latest=owners[-1] if owners else {}
    pilot=OUT/'05_backups/put_seed20269730/XVLA_capture_v2'
    progress=optional(pilot/'recording_progress.json',{})
    packaged=list(OUT.glob('*/scene*/PAIR_AUDIT.json'))+list(OUT.glob('05_backups/backup*/PAIR_AUDIT.json'))
    certified=[]
    for path in packaged:
        review=optional(path.parent/'FINAL_VISUAL_REVIEW.json',{})
        if review.get('approved_for_main') is True and path.parent.parent.name!='05_backups':certified.append(path)
    main_media=[p for p in packaged if p.parent.parent.name!='05_backups']
    summary={'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'owner':latest,'CPU_candidates_audited':sum('actual_group' in r for r in ledger),
        'packaged_main_pairs':len(main_media),'manually_certified_main_pairs':len(certified),
        'manually_certified_principal_videos':2*len(certified),'requested_principal_videos':22,
        'all22_complete':len(certified)==11,'pilot_render_progress':progress,
        'principal_aliases_NOT_independent_trials':True}
    atomic_json(OUT/'LIVE_STATUS.json',summary)
    lines=['# R57 — paired PUT qualitative: trạng thái thực',
        '',f'Cập nhật UTC: {summary["updated_at_utc"]}.',
        '',f'- CPU audit: {summary["CPU_candidates_audited"]}/15 cặp đề xuất.',
        f'- Đã package/media-audit: {len(main_media)} cặp main; đã visual-review cuối và chứng nhận: {len(certified)}/11 cặp ({2*len(certified)}/22 video chính).',
        f'- Owner: `{latest.get("status","chưa chạy")}` / `{latest.get("stage","n/a")}`.',
        f'- Pilot HD: {progress.get("frames",0)}/600 frame nguồn; 6 wide + hand-eye cùng trạng thái. File đang encode chưa được coi là hoàn thành.',
        '', '## Provenance và giới hạn', '',
        '- Phía folder RL là scripted privileged feedback Oracle, KHÔNG phải policy RL đã học. Controller faults được công khai từng scene.',
        '- X-VLA step014500: replay vật lý action neural thật, kiểm exact action/state/RGB/input hashes. Không phải inference neural mới.',
        '- X-VLA chỉ có terminal release supervisor giữ pose/mở kẹp khi đã aligned; không sửa approach/carry/XYZ. Không tính thành VLA thuần hay cập nhật SR quantitative.',
        '- Chọn theo outcome để minh họa qualitative, không dùng sample chọn lọc này suy ra SR hoặc kết luận VLA yếu hơn RL nói chung.',
        '', '## Pilot failure có bằng chứng trước release', '',
        'Seed20269730: contact12,703N khi chưa có verified grasp; vật dịch26,53cm trong250ms, đạt1,725m/s. Không có put hoàn chỉnh. Oracle trên đúng initial scene hoàn thành put thật.',
        'Đã mở các ảnh thật trước/contact/after/terminal. Không gọi là in-transit drop vì chưa gắp được; không kết luận misidentification hay memory loss. Full encoded-video review vẫn bắt buộc.',
        '', '[Audit phản chứng pilot](audit/PILOT_IMPACT_REVIEW.md) · [Kế hoạch](PLAN.md) · [Ledger](candidate_ledger.json)',
        '', '## Candidate ledger', '',
        '|Seed|Oracle fault|VLA full PUT|Oracle full PUT|Gate mạnh trước release|Trạng thái|',
        '|---|---|---|---|---|---|']
    for row in ledger:
        lines.append('|'+ '|'.join(str(row.get(k,'pending')).replace('|','/') for k in (
            'seed','oracle_fault','VLA_full_put','Oracle_full_put','VLA_failure_has_measured_pre_release_event','status'))+'|')
    lines+=['', '## Cleanup và training', '',
        'Đã gộp1110 file MP4 byte-identical bằng hardlink, giải phóng3.10GiB allocated bytes; giữ mọi pathname/content/SHA. Không xóa data/weights/checkpoint hoặc footage độc nhất. Xem cleanup_manifest.json.',
        'Các optimizer/evaluator R56 không bị dừng hoặc sửa; qualitative dùng một CPU child và một worker HD, admission theo RAM/VRAM.',
        'Không coi MISSING/backup/manual-review-pending là PASS. Bộ này chưa hoàn thành đủ22 video.']
    report='\n'.join(lines)+'\n';temp=OUT/'REPORT.md.tmp';temp.write_text(report);temp.replace(OUT/'REPORT.md')
    blocks=['<!doctype html><meta charset="utf-8"><title>R57 PUT qualitative — live audit</title>',
        '<style>body{font:16px sans-serif;max-width:1180px;margin:24px auto;background:#f5f7f9;color:#18212a}img{max-width:100%}video{width:100%}section{background:white;padding:18px;margin:18px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>',
        '<h1>PUT qualitative — scripted Oracle / X-VLA + terminal release</h1>',
        f'<p>{2*len(certified)}/22 principal videos certified. Pending media/visual gates are NOT completion.</p>',
        '<p>Oracle is not learned RL. Outcome-selected qualitative cases are not quantitative success rates. X-VLA failures are not scripted; any Oracle fault and terminal release supervisor are disclosed.</p>',
        '<p><a href="REPORT.md">Report</a> · <a href="PLAN.md">Plan</a> · <a href="LIVE_STATUS.json">Live status</a> · <a href="candidate_ledger.json">All candidates</a></p>',
        '<section><h2>Measured pilot: impact → object displacement → failed PUT</h2><p>Seed20269730, native X-VLA actions; no release supervisor activated. Contact12.703N; displacement26.53cm/250ms. Images below are actual physical-state previews, not animation or completed media certification.</p>']
    for frame in (266,271,280,599):
        rel=f'05_backups/put_seed20269730/XVLA_capture_v2/event_previews/frame_{frame:05d}.jpg'
        if (OUT/rel).exists():blocks.append(f'<figure><img loading="lazy" src="{rel}"><figcaption>Source frame {frame}, physical time {(frame+1)/30:.3f}s.</figcaption></figure>')
    blocks.append('</section><h2>Packaged paired videos (final review status explicit)</h2>')
    for path in packaged:
        rel=path.parent.relative_to(OUT)/'index.html';blocks.append(f'<p><a href="{rel}">{html.escape(str(path.parent.relative_to(OUT)))}</a> — media decoded; final visual review status in pair folder.</p>')
    if not packaged:blocks.append('<p>No full paired package yet. HD rendering and CPU candidate audits running; do not use incomplete MP4s.</p>')
    temp=OUT/'index.html.tmp';temp.write_text('\n'.join(blocks));temp.replace(OUT/'index.html')
    print({k:v for k,v in summary.items() if k not in ('owner',)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    a=parser.parse_args();deadline=time.monotonic()+36*3600
    while True:
        publish()
        if not a.watch or time.monotonic()>deadline:break
        time.sleep(30)
