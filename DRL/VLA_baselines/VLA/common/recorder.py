"""Single-video recorder used by every benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .observation import VLAObservation


class VideoRecorder:
    def __init__(self, output_path: Path, fps: int = 10, overview_render=None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
        self.overview_render = overview_render

    def add(self, observation: VLAObservation, label: str, repeat: int = 1) -> None:
        front = observation.front_rgb
        wrist = observation.wrist_rgb
        views = [front, wrist]
        if self.overview_render is not None:
            views.insert(0, self.overview_render())  # video only, NEVER policy input
        frame = np.concatenate(views, axis=1)
        if self.overview_render is not None:
            # Keep the human-readable label out of the overview pixels so a
            # high initial arm is not hidden by the video banner itself.
            frame = np.concatenate((np.zeros((32, frame.shape[1], 3), dtype=np.uint8), frame), axis=0)
        frame = np.ascontiguousarray(frame.copy())
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 31), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label[:100],
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labels = ("overview (video only)", "front (policy)", "wrist (policy)") if len(views) == 3 else ("front", "wrist")
        for number, name in enumerate(labels):
            cv2.putText(frame, name, (number*front.shape[1]+8, frame.shape[0]-9),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        for _ in range(repeat):
            self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class NullVideoRecorder:
    """Recorder-compatible sink for development sweeps.

    Final benchmark runs still use :class:`VideoRecorder`; this sink only
    avoids spending CPU time and disk space encoding every hyperparameter
    selection rollout.  It deliberately keeps the same context-manager and
    ``add`` interface so observation generation and policy timing are
    unchanged.
    """

    def add(self, observation: VLAObservation, label: str, repeat: int = 1) -> None:
        del observation, label, repeat

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullVideoRecorder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
