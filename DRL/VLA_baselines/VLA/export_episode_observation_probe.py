"""Export immutable actual observations at specified diagnostic timestamps."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from VLA.audit_native_camera_tensors import transformer
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    row = next(r for r in json.loads((args.source/"index.json").read_text()) if r["file"] == args.episode)
    payload = args.source/args.episode
    digest = sha256_file(payload)
    convert, method = transformer(SimpleNamespace(model="smolvla", checkpoint=args.checkpoint))
    args.output.mkdir(parents=True)
    sheet = Image.new("RGB", (512*2, 550*len(args.frames)), "white")
    draw = ImageDraw.Draw(sheet)
    with np.load(payload, allow_pickle=False) as arrays:
        for i, t in enumerate(args.frames):
            if not 0 <= t < row["steps"]:
                raise ValueError("Requested probe frame outside trajectory")
            for j, camera in enumerate(("front", "wrist")):
                _, native, mask = convert(arrays[camera+"_rgb"][t], arrays[camera+"_place_mask"][t])[0]
                path = args.output/f"frame{t}_{camera}_native512.png"
                Image.fromarray(native).save(path)
                sheet.paste(Image.fromarray(native), (512*j, 550*i+38))
                draw.text((512*j+4, 550*i+3), f"t={t} {camera} native goal pixels={int(mask.sum())}", fill="black")
                draw.text((512*j+4, 550*i+18), row["task"]["prompt"], fill="black")
    sheet.save(args.output/"native512_probe_sheet.png")
    if sha256_file(payload) != digest:
        raise RuntimeError("Episode changed during probe")
    (args.output/"probe.json").write_text(json.dumps({"source": str(args.source.resolve()),
        "episode": args.episode, "task": row["task"], "payload_sha256": digest,
        "frames": args.frames, "method": method, "policy_inputs_changed": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
