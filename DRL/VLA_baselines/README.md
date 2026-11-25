## VLA baselines

Source-only PyBullet adapters, demonstration processing, fine-tuning and evaluation
code for **FLOWER, X-VLA, π₀, VLA-Adapter, SmolVLA, VLA-JEPA, Dita, VLANeXt-S,
FALCON-ESM and Seer**. Older model integrations and regression tests are retained
where needed by the existing workflows. Model links, exact local upstream
revisions and entry-point locations are in [MODELS.json](MODELS.json) and
[UPSTREAMS.json](UPSTREAMS.json).

- `VLA/`: original Python namespace, shared environment, model adapters and recipes.
- `tools/`: source verification, optional asset linking and upstream restoration.

**Not included:** weights, datasets, experiment outputs, videos, caches or virtual
environments. Run from this directory with `PYTHONPATH=.`. Use separate environments
and installation instructions for each upstream model; this is not a single
pip-installable training pipeline. Historical scripts contain machine-specific
paths and artifact checks: configure these and supply the matching data/weights
before training. Restoring source alone does not reproduce a result.

```bash
python tools/verify_source.py
python tools/link_assets.py --drl-root ..   # reuse this DRL checkout's asset/
python tools/restore_upstreams.py --list    # show pinned source dependencies
# Optional, downloads source but not model weights:
# python tools/restore_upstreams.py --fetch flower_vla_pret
```

The shared executor includes IK servo, safety checks and a contact-qualified grasp
constraint. Keep these and the chosen observation/action protocol explicit when
reporting results. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing.
