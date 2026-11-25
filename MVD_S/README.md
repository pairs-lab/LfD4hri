# MVD-S action-recognition branch

Native **MVD ViT-S/16** (21,946,926 parameters with the 174-class head), separate
from the TSM/MMAction implementation. It consumes RGB video and returns the same
action strings used by the existing hand/object/keyframe/VLM pipeline.
Source: [MVD, CVPR 2023](https://github.com/ruiwang2021/mvd), pinned in
[`UPSTREAM.json`](UPSTREAM.json). Keep the upstream non-commercial license and
notices in `vendor/` when redistributing this code.

## Install and infer

Use a dedicated environment; this branch does not require MMCV/MMAction:

```bash
conda create -n lfd-mvd python=3.11
conda activate lfd-mvd
cd LfD4hri
pip install -r MVD_S/requirements.txt
python -m MVD_S --video /path/clip.mp4 \
  --checkpoint /path/mvd_s_ssv2_finetuned.pth --device cpu --preset paper
```

Supply a **fine-tuned Something-Something V2 checkpoint with 174 classes**, in
native MVD ViT-S format. Pretrain-only and Kinetics-400 heads are rejected rather
than silently replaced. [`labels.txt`](labels.txt) is the existing TSM task-string
map, copied without reordering; train with the standard SSV2 class IDs 0–173.
Only load trusted checkpoint files; optional `--sha256` verifies a known digest.

`paper` uses 16 frames × 2 temporal clips × 3 spatial crops, no flip, ImageNet
RGB normalization and **mean per-view softmax**, matching upstream test code.
`fast` is an explicitly different 16×1×1 centre-view preset. Use `--device cuda:0`
when CUDA works; views are processed one at a time by default. `--output new.json`
saves scores, checkpoint hash, decoder, sampled indices and crop metadata.

## Select MVD instead of TSM in the original pipeline

After installing the **original** detector/keyframe/LLaVA dependencies and
supplying their checkpoints, run:

```bash
python demo.py --single_video --video /path/clip.mp4 \
  --action_backend mvd_s --mvd_checkpoint /path/mvd_s_ssv2_finetuned.pth \
  --device cpu --mvd_run_dir MVD_S/runs/example01
```

The run directory must be new. Outputs are isolated there, including
`action_predictions.jsonl`; detector/VLM arguments otherwise remain unchanged.
`--action_backend tsm` is still the default in `demo.py`, `demo_wo_data.py`,
`demo_wo_key.py` and `demo_wo_os.py`. Without `--single_video`, the historical
`Dataset/Dataset` batch selection remains in effect. The full pipeline still
requires its non-action-model dependencies; the standalone command above does not.

## Train / fine-tune

Create separate `train.csv` and `val.csv`, one `video_path class_id` per line.
Paths can be relative to `--data-root`; IDs are integers 0–173. Video paths shared
between splits are rejected (also keep different recordings of the same source
out of the other split when preparing the lists).

```bash
python -m MVD_S.train \
  --train-list /data/train.csv --val-list /data/val.csv --data-root /data/videos \
  --pretrained /path/mvd_s_pretrained.pth \
  --output MVD_S/runs/finetune01 --device cuda:0 \
  --epochs 40 --batch-size 2 --accum-steps 8
```

This **practical supervised adapter recipe** trains the native video transformer
and classifier using AdamW (β=0.9/0.999, decay 0.05), layer decay 0.7,
CE label smoothing 0.1, temporal jitter and one shared random spatial crop.
Base LR 0.001 is scaled by effective batch/256; warmup is 5 epochs, followed by
cosine decay to 1% of peak. There is no horizontal flip. Validation uses `fast`.
`best.pth` is selected by validation top-1; `last.pth` and `TRAINING.json` are also
saved. Output directories cannot be overwritten. These are inference/weight
continuation checkpoints, **not exact optimizer/RNG-resume checkpoints**.

It is **not** the complete published MVD training recipe: no masked-distillation
pretraining, RandAugment, mixup, CutMix or distributed/repeated augmentation is
implemented here. For an exact paper reproduction use the
[official training instructions](https://github.com/ruiwang2021/mvd/blob/45b877a6d18202a81528eeb31ff2da2c0e9ef2ea/FINETUNE.md)
and [ViT-S script](https://github.com/ruiwang2021/mvd/blob/45b877a6d18202a81528eeb31ff2da2c0e9ef2ea/scripts/mvd_vit_small_from_vit_base_teacher_epoch_400/finetune_on_ssv2.sh).
The [model zoo](https://github.com/ruiwang2021/mvd/blob/45b877a6d18202a81528eeb31ff2da2c0e9ef2ea/MODEL_ZOO.md)
provides pretraining weights; do not treat those as a trained action head.

## Software checks

```bash
python -B -m unittest -v MVD_S.tests.test_backend
python -B -m MVD_S.smoke --output /path/to/new/smoke_test
```

The bounded smoke creates two synthetic clips, performs one real optimizer
update, verifies changed saved weights, then reloads and runs six-view inference
on CPU. Its checkpoints are marked `software_smoke_only` and rejected by normal
inference. This does not measure recognition quality or train a paper checkpoint.
