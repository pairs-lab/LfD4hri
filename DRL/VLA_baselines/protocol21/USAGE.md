# Robot-state adapters

Use `protocol21` as the entry point, not archived stage-specific servers/job owners
that refer to earlier datasets and experiment receipts.

## Inputs

Each method receives front/wrist RGB, the instruction and shared raw21: q6,
closure1, pad-centre XYZ3, previous signed-close1, tool rotation6, surface jaw gap1,
left/right generic external-contact bits2 and summed pad normal force1. Positions
are world-frame metres; rotation stacks the first two tool-to-world columns.
Contact sums exclude self contacts, not wrong objects/table. No object/goal
coordinates or scene-map image enter this interface.

`inputs.py`, `specs.py` and `state21_reference.py` define native prefixes,
normalization, history and H/K. FALCON requires synchronized metric front depth,
intrinsics and world-to-OpenCV-camera extrinsics. Seer has a separate native7 branch.
Fit statistics on unique TRAIN timesteps; D3 can reuse D1 robot21 statistics.
Old67 statistics and adaptations are incompatible.

## Data

Manifest: `schema: vla21-episodes-v1`, explicit `split` (TRAIN/VAL/TEST), and `rows`.
Each row contains `episode_id`, `scene_id`, `file`, payload `sha256`, `instruction`,
`task`, `source_kind` and `target_name` (sampling label, not input). Pass reserved
scene IDs to `Bank(..., reserved_scenes=...)` to reject scene-group overlap.

Each NPZ contains aligned `raw_state21[N,21]`, `ee_position[N,3]`, uint8 HWC streams
`front_rgb`/`wrist_rgb`, `canonical7[N,7]` and `time_s[N]` at 10 Hz. Log observation
before action. Actions are world XYZ deltas, local XYZ Euler deltas and signed-close
(-1 open, +1 close). The previous-command channel contains the preceding applied
gripper command, not the current label. FALCON also needs `depth_z_m[N,H,W]`,
`intrinsics[N,3,3]` and `extrinsics_OpenCV[N,4,4]`. Do not rename old target-filtered
contacts to generic contacts; record fresh data if generic feedback was not logged.

```bash
python -m protocol21.run audit --model xvla --manifest /data/train/index.json
python -m protocol21.run audit --model seer --manifest /data/val/index.json \
  --stats /data/train/stats.json
```

`Bank.statistics()` returns JSON-serializable statistics; `Bank.sample()` creates
native supervised samples. Future actions/images are label fields only.
`native.create(...).loss(sample)` uses native supervised objectives. No training
has been performed by these checks; earlier budgets are not already-completed
updates on the new state.

## Native inference

Install each upstream in its own environment. `--resources` selects a DRL-like
resource tree with `VLA/third_party/`, `VLA/checkpoints/` and required lightweight
metadata; `native.py` lists per-family locations. Restore pinned sources with
`tools/restore_upstreams.py`; supply foundation weights/tokenizers separately.

```bash
python -m protocol21.resources --resources /path/to/resources
```

```bash
python -m protocol21.run native --model xvla --manifest /data/train/index.json \
  --stats /data/train/stats.json --resources /path/to/resources --device cuda
```

`--foundation` overrides a foundation path where supported. π₀ needs an OpenPI
checkpoint with `params/` and `assets/`; other factories adapt released native
foundations. A new state projector is not a trained robot21 policy. Save/load
trainables with `checkpoint.py`; use `--adaptation /path/to/robot21_checkpoint` to
reload them (exact family, schema, stats and tensor shapes checked). Missing CUDA,
weights or dependencies produce an error, not substitute actions.

For online execution use `Executor.step(frame)` at every 10 Hz control tick.
H/K is preserved; X-VLA decodes against the measured pose at each executed step,
Seer against the preceding requested pose. No task-coordinate action repair is added.
After `env.step(action)`, call `Executor.record_execution(info)` so the preceding
requested pose includes the shared safety controller's actual action clamps.
