21-D contract and inherited numerical recipes, not completed model adapters.

state21_reference.py constructs/normalizes only the shared numerical input.
The ten historical runners in ../VLA/ have NOT all been migrated to this schema.
Their previous checkpoints/results must not be described as 21-D experiments.

vla21-lock-v1.json, camera_profile.json and the TeX fragments describe the new
protocol. References to recipe_sources/ in the historical lock correspond to
the original source tree; the matching recipes are exported under ../VLA/.
Indexed data payloads, historical receipts and dataset indices are not included.
The lock is preserved unchanged as a protocol reference, not a runtime receipt.

Test: python -B -m unittest -v test_state21_reference.py (requires numpy).
