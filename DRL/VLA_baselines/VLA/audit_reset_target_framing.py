"""Immutable reset edge-framing diagnostic; NOT a new training/success gate.

Segmentation is used only for audit. A mask touching the image boundary is
a clipping risk, not proof that a policy cannot recognize the object. Missing
pixels are not interpreted as an unclipped view. No frames/scenes are removed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint
from VLA.common.spatial import FIXED_DESTINATION


def framing(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Expected a real 2-D boolean audit mask")
    yy, xx = np.where(mask)
    if not len(yy):
        return {"pixels": 0, "bbox": [0, 0], "border_margin": -1,
                "edge_risk": False, "visible_not_at_edge": False}
    margin = int(min(yy.min(), xx.min(), mask.shape[0]-1-yy.max(), mask.shape[1]-1-xx.max()))
    return {"pixels": int(len(yy)), "bbox": [int(np.ptp(xx)+1), int(np.ptp(yy)+1)],
            "border_margin": margin, "edge_risk": margin == 0,
            "visible_not_at_edge": len(yy) >= 25 and margin >= 1}


def summarize(views):
    return {"any_visible_25px": any(v["pixels"] >= 25 for v in views.values()),
            "any_visible_not_at_edge": any(v["visible_not_at_edge"] for v in views.values()),
            "visible_edge_risk_without_interior_alternative":
                any(v["pixels"] >= 25 and v["edge_risk"] for v in views.values())
                and not any(v["visible_not_at_edge"] for v in views.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    state = source / "collection_state.json"
    if state.exists() and not json.loads(state.read_text())["status"].startswith(("complete", "completed")):
        raise ValueError("Requires complete source, not a moving collection prefix")
    bound = {str(p): sha256_file(p) for p in (source/"index.json", source/"dataset_manifest.json", Path(__file__))}
    index = json.loads((source/"index.json").read_text())
    if not index or len({r["file"] for r in index}) != len(index):
        raise ValueError("Empty/duplicate source index")
    payload = source_payload_fingerprint(source, index)
    rows, counts = [], Counter()
    for item in index:
        physical_goal = item["task"].get("place_name") not in (None, *FIXED_DESTINATION)
        with np.load(source/item["file"], allow_pickle=False) as data:
            for role in (("pick", "place") if physical_goal else ("pick",)):
                views = {cam: framing(data[cam+"_"+role+"_mask"][0]) for cam in ("front", "wrist")}
                flags = summarize(views)
                rows.append({"file": item["file"], "task": item["task"]["name"],
                             "target": item["task"][role+"_name"], "role": role,
                             "views": views, **flags})
                for flag, value in flags.items():
                    counts[role+":"+flag] += int(value)
                counts[role+":samples"] += 1
    # Select only from acquisition reset metrics; no benchmark/outcome tuning.
    selected = []
    for task in ("touch", "pick", "move", "pick_place"):
        candidates = [r for r in rows if r["task"] == task and r["role"] == "pick"]
        if not candidates:
            raise ValueError("All four primitives are required")
        selected.append(min(candidates, key=lambda r: (
            r["any_visible_not_at_edge"],
            max(v["border_margin"] for v in r["views"].values()), r["file"])))
    output.mkdir(parents=True)
    sheet = Image.new("RGB", (768, 408*4), "black")
    for number, row in enumerate(selected):
        with np.load(source/row["file"], allow_pickle=False) as data:
            for k, cam in enumerate(("front", "wrist")):
                rgb = Image.fromarray(data[cam+"_rgb"][0])
                rgb.save(output/(Path(row["file"]).stem+"_reset_"+cam+".png"))
                sheet.paste(rgb.resize((384, 384)), (384*k, 408*number+24))
        ImageDraw.Draw(sheet).text((3, 408*number+4), row["task"]+" "+row["target"]+" | "+row["file"], fill="white")
    sheet.save(output/"reset_framing_contact_sheet.png")
    verify_source_payload_fingerprint(payload, source, index)
    if any(sha256_file(Path(p)) != digest for p, digest in bound.items()):
        raise RuntimeError("Source/code changed during framing audit")
    report = {"scope": "Full-source reset edge-framing DIAGNOSTIC, not full silhouette visibility, training qualification or learned SR",
              "source": str(source), "bound_files": bound, "source_payload_fingerprint": payload,
              "episodes": len(index), "counts": dict(counts), "samples": rows, "exported_samples": selected,
              "policy_inputs_changed": False, "training_authorized": False, "final_test_used": False,
              "caveat": "Interior visible masks can still be occluded by the arm; edge risk does not prove unrecognizability"}
    (output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"episodes": len(index), "counts": dict(counts), "contact_sheet": str(output/"reset_framing_contact_sheet.png")}))


if __name__ == "__main__":
    main()
