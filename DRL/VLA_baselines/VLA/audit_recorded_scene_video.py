"""Extract real overview/policy-camera video frames for human audit, no edits."""
import argparse
import json
from pathlib import Path

import cv2

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json


def extract(trial, output):
    if output.exists(): raise FileExistsError("Keep earlier video inspection artifacts")
    result = json.loads((trial / "results.json").read_text())
    if result.get("integration_error") or len(result["tasks"]) != 4:
        raise ValueError("A completed four-task video is required")
    video = trial / "rollout.mp4"
    digest = sha256_file(video)
    reader = cv2.VideoCapture(str(video))
    try:
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(reader.get(cv2.CAP_PROP_FPS))
        expected = sum(t["steps"] + 20 for t in result["tasks"])
        if not reader.isOpened() or count != expected or abs(fps - 10) > 1e-6:
            raise ValueError("Video frame count/FPS differ from actual rollouts")
        output.mkdir(parents=True)
        rows, offset = [], 0
        for task in result["tasks"]:
            for phase, frame in (("reset", offset), ("terminal", offset + 10 + task["steps"] - 1)):
                reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = reader.read()
                if not ok: raise ValueError("Cannot decode the recorded video frame")
                path = output / f"{task['name']}_{phase}.png"
                if not cv2.imwrite(str(path), pixels): raise ValueError("Video snapshot export failed")
                rows.append({"task": task["name"], "phase": phase, "frame": frame,
                    "seconds": frame / fps, "path": str(path), "sha256": sha256_file(path),
                    "shape": list(pixels.shape)})
            offset += task["steps"] + 20
    finally:
        reader.release()
    if sha256_file(video) != digest: raise ValueError("Video changed during extraction")
    report = {"passed_decode_integrity": True, "source_video": str(video), "video_sha256": digest,
        "decoded_selected_frames": len(rows), "actual_frame_count": count, "fps": fps, "rows": rows,
        "model_success_certified": False, "all_video_frames_visually_reviewed": False,
        "overview_is_VIDEO_ONLY_not_policy_input": True}
    atomic_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.trial.resolve(), args.output.resolve()), indent=2))
