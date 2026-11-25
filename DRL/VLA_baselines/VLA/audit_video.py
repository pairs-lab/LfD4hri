#!/usr/bin/env python3
"""Export start/middle/end frames of each task from an actual rollout video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    fps = int(manifest["control_hz"])
    tasks = result["tasks"]
    column_width = 768 if manifest.get("video_only_overview") else 512
    sheet = Image.new("RGB", (column_width * 3, 280 * len(tasks)), "black")
    reader = imageio.get_reader(args.run / "rollout.mp4")
    offset = 0
    try:
        for row, task in enumerate(tasks):
            steps = int(task["steps"])
            for col, (phase, frame) in enumerate((
                ("reset", offset), ("middle", offset + fps + steps // 2),
                ("end", offset + fps + max(steps - 1, 0)),
            )):
                image = Image.fromarray(reader.get_data(frame)).resize((column_width, 256))
                sheet.paste(image, (col * column_width, row * 280 + 24))
                ImageDraw.Draw(sheet).text(
                    (col * column_width + 4, row * 280 + 5),
                    f"{task['name']} {phase} | success={task['success']} | video frame={frame}",
                    fill="white",
                )
            offset += steps + 2 * fps
    finally:
        reader.close()
    output = args.run / "video_contact_sheet.png"
    sheet.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
