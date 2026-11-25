"""Extract actual decoded frames from ALL completed four-task rollout videos.

No regenerated simulation/AI images or outcome-based episode filtering. The
side-by-side video includes an overview for humans, never as policy input.
Frame positions follow the recorder's pre-action frames plus terminal frame.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from VLA.common.provenance import sha256_file


def frame_plan(results):
    offset, rows = 0, []
    for trial in results:
        steps = int(trial["steps"])
        if steps < 1 or len(trial["trace"]) != steps:
            raise ValueError("Video mapping requires every recorded action step")
        frames = sorted(set([0, min(5, steps), steps//2, max(0, steps-1), steps]))
        rows.append({"file": trial["file"], "prompt": trial["prompt"],
                     "task": trial["task"], "success": trial["success"],
                     "video_frame_offset": offset,
                     "relative_frames": frames, "absolute_frames": [offset+i for i in frames]})
        offset += steps+1
    return offset, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    reports = []
    for report_path in args.results:
        results = json.loads(report_path.read_text())
        video = report_path.parent/"rollout.mp4"
        expected_count, trials = frame_plan(results["results"])
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError("Cannot decode actual rollout video: "+str(video))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != expected_count or capture.get(cv2.CAP_PROP_FPS) != 10:
            raise ValueError(f"Recorder mapping mismatch: {count} != {expected_count}")
        split = results["split_mode"]
        folder = args.output/split
        folder.mkdir()
        for trial in trials:
            panels = []
            for relative, absolute in zip(trial["relative_frames"], trial["absolute_frames"]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, absolute)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Video frame cannot be decoded")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                filename = f"{Path(trial['file']).stem}_frame{relative:04d}.png"
                Image.fromarray(rgb).save(folder/filename)
                panel = Image.new("RGB", (576, 232), "white")
                panel.paste(Image.fromarray(rgb).resize((576, 208)), (0, 24))
                ImageDraw.Draw(panel).text((4, 4), f"actual video | step {relative} | {trial['prompt']}", fill="black")
                panels.append(panel)
            sheet = Image.new("RGB", (576, 232*len(panels)), "white")
            for number, panel in enumerate(panels):
                sheet.paste(panel, (0, number*232))
            sheet.save(folder/f"{Path(trial['file']).stem}_actual_video_sequence.png")
        capture.release()
        reports.append({"results": str(report_path.resolve()), "results_sha256": sha256_file(report_path),
            "video": str(video.resolve()), "video_sha256": sha256_file(video), "decoded_frame_count": count,
            "trials": trials, "all_completed_trials_included": True})
    (args.output/"audit.json").write_text(json.dumps({"scope": "actual_decoded_rollout_video_frames_NOT_manual_review_certificate",
        "reports": reports, "simulation_rerendered": False, "model_inputs_modified": False,
        "script_sha256": sha256_file(Path(__file__))}, indent=2)+"\n")


if __name__ == "__main__":
    main()
