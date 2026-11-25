"""Export true image-preprocessor outputs; never load neural policy weights.

Run each model in its own venv. These previews are camera/data evidence,
NOT trained inference, full input parity or a dataset-sufficiency certificate.
Audit masks pass through identical geometry only to measure retained pixels.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.provenance import sha256_file


def transformer(args):
    if args.model == "pi0":
        from openpi_client.image_tools import resize_with_pad
        def transform(rgb, mask):
            image = np.asarray(resize_with_pad(rgb, 224, 224))
            resized_mask = np.asarray(resize_with_pad(np.repeat((mask*255).astype(np.uint8)[..., None], 3, -1), 224, 224))[..., 0] >= 128
            return [("openpi_224", image, resized_mask)]
        return transform, "official openpi_client.image_tools.resize_with_pad (inference image geometry)"
    import torch
    torch.set_num_threads(1)
    if args.model == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        if args.checkpoint is None:
            raise ValueError("SmolVLA requires --checkpoint to inspect the actual saved image configuration")
        cfg = SmolVLAConfig.from_pretrained(args.checkpoint)
        features = list(cfg.image_features)
        if set(features) not in ({"observation.images.front", "observation.images.wrist"},
                                 {"observation.images.camera1", "observation.images.camera2"}):
            raise ValueError(f"Unexpected camera features: {features}")
        view = SimpleNamespace(config=cfg)
        def transform(rgb, mask):
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
            mask_tensor = torch.from_numpy(mask.copy()).float()[None, None].expand(1, 3, *mask.shape)
            images, present = SmolVLAPolicy.prepare_images(view, {features[0]: tensor, features[1]: tensor})
            masks, _ = SmolVLAPolicy.prepare_images(view, {features[0]: mask_tensor, features[1]: mask_tensor})
            if not bool(present[0].all()):
                raise ValueError("A real camera was padded away")
            if not torch.isfinite(images[0]).all():
                raise ValueError("Non-finite native SmolVLA pixels")
            image = ((images[0][0]+1)*127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            return [("smol_official_prepare_images", image, masks[0][0, 0].numpy() >= 0)]
        return transform, "official SmolVLAPolicy.prepare_images with saved config; image-only, not old checkpoint state normalization"
    if args.model == "rdt":
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.processor or "google/siglip-so400m-patch14-384")
        def prepare(array):
            return processor(images=Image.fromarray(array), return_tensors="pt")["pixel_values"][0]
        def transform(rgb, mask):
            normalized = prepare(rgb)
            if not torch.isfinite(normalized).all():
                raise ValueError("Non-finite native RDT pixels")
            mask_normalized = prepare(np.repeat((mask*255).astype(np.uint8)[..., None], 3, -1))
            mean = np.asarray(processor.image_mean)[:, None, None]
            std = np.asarray(processor.image_std)[:, None, None]
            image = np.rint(np.clip(normalized.numpy()*std+mean, 0, 1)*255).astype(np.uint8).transpose(1, 2, 0)
            resized_mask = (mask_normalized.numpy()*std+mean)[0] >= .5
            return [("rdt_siglip_384", image, resized_mask)]
        return transform, "actual RDT SigLIP AutoImageProcessor (image-only, not T5/history/state tensors)"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor or "openvla/openvla-7b", trust_remote_code=True).image_processor
    def transform(rgb, mask):
        pixels = processor.apply_transform(Image.fromarray(rgb))
        if not torch.isfinite(pixels).all():
            raise ValueError("Non-finite native OpenVLA pixels")
        mask_pixels = processor.apply_transform(Image.fromarray(np.repeat((mask*255).astype(np.uint8)[..., None], 3, -1)))
        result = []
        for k, name in enumerate(("openvla_dino", "openvla_siglip")):
            mean, std = np.asarray(processor.means[k])[:, None, None], np.asarray(processor.stds[k])[:, None, None]
            restored = pixels[k*3:k*3+3].numpy()*std+mean
            resized_mask = (mask_pixels[k*3:k*3+3].numpy()*std+mean)[0] >= .5
            result.append((name, np.rint(np.clip(restored, 0, 1)*255).astype(np.uint8).transpose(1, 2, 0), resized_mask))
        return result
    return transform, "actual OpenVLA apply_transform, both DINO and SigLIP backbones; front-only native policy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", choices=("pi0", "smolvla", "rdt", "openvla"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--processor")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state_path = args.source / "collection_state.json"
    if state_path.exists() and not json.loads(state_path.read_text())["status"].startswith(("completed", "complete")):
        raise ValueError("Wait for complete collection before immutable native previews")
    initial_index_hash = sha256_file(args.source/"index.json")
    index = json.loads((args.source / "index.json").read_text())
    selected = []
    selected_files = set()
    for name in ("touch", "pick", "move", "pick_place"):
        scored = {camera: [] for camera in (("front",) if args.model == "openvla" else ("front", "wrist"))}
        for item in index:
            if item["task"]["name"] == name:
                with np.load(args.source / item["file"], allow_pickle=False) as episode:
                    for camera in scored:
                        scored[camera].append((int(episode[camera+"_pick_mask"][0].sum()), item))
        if not all(scored.values()):
            raise ValueError(f"Missing task {name}; wait for complete collection")
        for values in scored.values():
            item = min(values, key=lambda x:x[0])[1]
            if item["file"] not in selected_files:
                selected.append(item)
                selected_files.add(item["file"])
    convert, method = transformer(args)
    args.output.mkdir(parents=True)
    rows, cells = [], []
    for item in selected:
        with np.load(args.source / item["file"], allow_pickle=False) as episode:
            steps = [0, len(episode["canonical_actions"])//2, len(episode["canonical_actions"])-1]
            for phase, step in zip(("reset", "middle", "final_expert"), steps):
                images = []
                for camera in (("front",) if args.model == "openvla" else ("front", "wrist")):
                    for backbone, rgb, mask in convert(episode[camera+"_rgb"][step], episode[camera+"_pick_mask"][step]):
                        stem = f"{Path(item['file']).stem}_{phase}_{camera}_{backbone}"
                        Image.fromarray(rgb).save(args.output / (stem+".png"))
                        yy, xx = np.where(mask)
                        rows.append({"episode": item["file"], "task": item["task"]["name"], "target": item["task"]["pick_name"],
                                     "phase": phase, "step": step, "camera": camera, "backbone": backbone,
                                     "native_size": list(rgb.shape), "target_pixels": int(mask.sum()),
                                     "target_bbox": [int(np.ptp(xx)+1), int(np.ptp(yy)+1)] if len(xx) else [0, 0],
                                     "finite": bool(np.isfinite(rgb).all()), "png": stem+".png"})
                        images.append(Image.fromarray(rgb).resize((224, 224)))
                cell = Image.new("RGB", (224*len(images), 248), "black")
                for k, image in enumerate(images):
                    cell.paste(image, (224*k, 24))
                ImageDraw.Draw(cell).text((3, 4), f"{item['task']['name']} {item['task']['pick_name']} {phase}", fill="white")
                cells.append(cell)
    sheet = Image.new("RGB", (cells[0].width*3, 248*((len(cells)+2)//3)), "black")
    for k, cell in enumerate(cells):
        sheet.paste(cell, ((k%3)*cell.width, (k//3)*248))
    sheet.save(args.output / "native_inputs_contact_sheet.png")
    if sha256_file(args.source/"index.json") != initial_index_hash:
        raise RuntimeError("Source metadata changed during native preview")
    report = {"scope": "Actual image preprocessor preview, not learned inference/full sensor parity or data sufficiency",
              "model": args.model, "source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source/"index.json"),
              "method": method, "neural_weights_loaded": False, "segmentation_given_to_policy": False,
              "selection": "smallest reset target in front (and wrist for multi-camera models), deduplicated per task",
              "final_test_used": False, "samples": rows}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"model": args.model, "frames": len(rows), "contact_sheet": str(args.output / "native_inputs_contact_sheet.png")}))


if __name__ == "__main__":
    main()
