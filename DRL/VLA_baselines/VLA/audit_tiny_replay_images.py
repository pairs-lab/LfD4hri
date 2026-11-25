"""Assemble saved policy observations, not overview/video-only camera images."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.replay / "results.json").read_text())
    rows = report["results"]
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    sheet = Image.new("RGB", (896, 248 * len(rows)), "black")
    metadata = []
    for index, row in enumerate(rows):
        folder = args.replay / Path(row["file"]).stem
        frames = sorted(folder.glob("step_*_front.png"))
        # Last saved observation can precede success; label the exact step.
        terminal = sorted(folder.glob("terminal_*_front.png"))
        for column, front in enumerate((frames[0], terminal[-1] if terminal else frames[-1])):
            step = int(front.name.split("_")[1])
            wrist = front.with_name(front.name.replace("_front", "_wrist"))
            cell = Image.new("RGB", (448, 248), "black")
            for camera_index, path in enumerate((front, wrist)):
                with Image.open(path) as image:
                    cell.paste(image.resize((224, 224)), (camera_index * 224, 24))
            ImageDraw.Draw(cell).text((3, 4),
                f"{row['task']} step {step} result={row['success']} | front/wrist", fill="white")
            sheet.paste(cell, (column * 448, index * 248))
            metadata.append({"prompt": row["prompt"], "step": step,
                "paths": [str(front.resolve()), str(wrist.resolve())], "success": row["success"],
                "is_terminal_observation": front.name.startswith("terminal_")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"scope": "Saved learned policy RGB, visual-review artifact only",
        "no_visibility_pass_claim": True, "final_test_used": False, "frames": metadata}, indent=2) + "\n")


if __name__ == "__main__":
    main()
