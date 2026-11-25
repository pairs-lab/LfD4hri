"""Visual audit of actual OpenVLA's two vision-backbone tensors (CPU only)."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoProcessor

from VLA.audit_privileged_native_inputs import SNAPSHOT
from VLA.openvla.audit_model_inputs import unnormalize


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text()); processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
    selected = []
    for task in ("touch", "pick", "move", "pick_place"):
        selected.extend([r for r in audit["samples"] if r["task"] == task][:3])
    sheet = Image.new("RGB", (448 * 3, 248 * 4), "black")
    for i, row in enumerate(selected):
        transform = processor.image_processor; tensor = transform.apply_transform(Image.open(row["source_front"]).convert("RGB"))
        cell = Image.new("RGB", (448, 248), "black"); draw = ImageDraw.Draw(cell)
        draw.text((3, 3), f"{row['task']} idx{row['index']} | DINO / SigLIP", fill="white")
        cell.paste(unnormalize(tensor[:3], transform.means[0], transform.stds[0]), (0, 24))
        cell.paste(unnormalize(tensor[3:], transform.means[1], transform.stds[1]), (224, 24))
        sheet.paste(cell, ((i % 3) * 448, (i // 3) * 248))
    sheet.save(args.output)


if __name__ == "__main__": main()
