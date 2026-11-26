## VLA baselines

Source-only PyBullet adapters, demonstration processing, fine-tuning and evaluation
code for **FLOWER, X-VLA, π₀, VLA-Adapter, SmolVLA, VLA-JEPA, Dita, VLANeXt-S,
FALCON-ESM and Seer**. Model links and upstream revisions are in
[MODELS.json](MODELS.json) and [UPSTREAMS.json](UPSTREAMS.json).

Run here with `PYTHONPATH=.` and each model's dependencies. Weights, datasets,
videos and virtual environments are not included.

```bash
python tools/link_assets.py --drl-root ..
python tools/verify_source.py
python -m protocol21.run smoke --steps 3
python tools/restore_upstreams.py --list
```

The smoke check exercises simulation/input interfaces, not learned-policy inference.
See [usage](protocol21/USAGE.md) for data and native inference commands.
The shared executor uses IK servo, safety checks and a contact-qualified grasp
constraint. Licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
