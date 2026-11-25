"""SSV2 temporal sampling and RGB views, matched to pinned upstream ssv2.py."""
from pathlib import Path
import math
import numpy as np
import torch
from .vendor.functional import normalize, resize_clip
from .vendor.volume_transforms import ClipToTensor

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def temporal_indices(length, *, clips=2, segments=16, rng=None):
    if length < 1 or segments != 16 or clips not in (1, 2):
        raise ValueError('Require a nonempty video, 16 segments and one/two clips')
    tick = float(length - 1) / segments
    result = []
    for clip in range(clips):
        indices = []
        for i in range(segments):
            if rng is not None:
                start, end = int(np.round(tick*i)), int(np.round(tick*(i+1)))
                index = int(rng.integers(start, end+1))
            else:
                start = int(np.round(tick*i + tick/clips*clip))
                end = int(np.round(tick*i + tick/clips*(clip+1)))
                index = (start+end)//2
            indices.append(min(max(index, 0), length-1))
        result.append(indices)
    return result


def decode_samples(video, *, clips=2, decoder='decord', rng=None):
    path = Path(video).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    if decoder == 'decord':
        from decord import VideoReader, cpu
        try:
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            indices = temporal_indices(len(reader), clips=clips, rng=rng)
            unique = sorted(set(x for row in indices for x in row))
            frames = reader.get_batch(unique).asnumpy()  # Decord returns RGB.
            fps, length = float(reader.get_avg_fps()), len(reader)
        except Exception as exc:
            raise ValueError(f'Cannot decode {path}: {exc}') from exc
    elif decoder == 'opencv':
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            indices = temporal_indices(length, clips=clips, rng=rng)
            unique = sorted(set(x for row in indices for x in row))
            decoded = []
            for index in unique:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode frame {index} of {path}')
                decoded.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames = np.stack(decoded)
        finally:
            cap.release()
    else:
        raise ValueError('Decoder must be decord or opencv')
    lookup = dict(zip(unique, frames))
    samples = [np.stack([lookup[i] for i in row]) for row in indices]
    return samples, {'video':str(path), 'frame_count':length,
                     'fps':fps if math.isfinite(fps) and fps > 0 else None,
                     'frame_indices':indices, 'decoder':decoder}


def make_views(samples, *, crops=3):
    if crops not in (1, 3):
        raise ValueError('Use one centre crop or three spatial crops')
    views, records = [], []
    for clip, frames in enumerate(samples):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != 16 or frames.shape[-1] != 3:
            raise ValueError('Expected 16 uint8 RGB frames (T,H,W,3)')
        # Upstream test Resize defaults to nearest, unlike validation bilinear.
        resized = np.stack(resize_clip(frames, 224, interpolation='nearest'))
        h, w = resized.shape[1:3]
        positions = [int(i*(max(h,w)-224)/2) for i in range(3)] if crops == 3 else [(max(h,w)-224)//2]
        for crop, offset in enumerate(positions):
            top, left = (offset, 0) if h >= w else (0, offset)
            images = resized[:, top:top+224, left:left+224, :]
            tensor = normalize(ClipToTensor()(images), MEAN, STD)
            if tensor.shape != (3,16,224,224) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid MVD input tensor')
            views.append(tensor)
            records.append({'temporal_clip':clip,'spatial_crop':crop,
                            'crop_xy':[left,top],'resized_hw':[h,w]})
    return views, records


def training_view(video, *, rng, decoder='decord'):
    """Practical supervised adapter recipe, not full upstream RandAugment/mixup."""
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    samples, _ = decode_samples(video, clips=1, decoder=decoder, rng=rng)
    tensor = ClipToTensor()(samples[0])  # C,T,H,W, float in [0,1].
    height, width = tensor.shape[-2:]
    top = left = 0
    crop_h, crop_w = height, width
    for _ in range(10):
        area = height*width*float(rng.uniform(.08,1.))
        ratio = math.exp(float(rng.uniform(math.log(.75),math.log(4/3))))
        candidate_w, candidate_h = int(round(math.sqrt(area*ratio))), int(round(math.sqrt(area/ratio)))
        if 0 < candidate_w <= width and 0 < candidate_h <= height:
            crop_h,crop_w = candidate_h,candidate_w
            top = int(rng.integers(0,height-crop_h+1));left = int(rng.integers(0,width-crop_w+1))
            break
    # Same spatial crop for every timestamp; no flip (SSV2 directional labels).
    frames = tensor.permute(1,0,2,3)
    frames = F.resized_crop(frames,top,left,crop_h,crop_w,[224,224],
                            interpolation=InterpolationMode.BICUBIC,antialias=True)
    return normalize(frames.permute(1,0,2,3), MEAN, STD)
