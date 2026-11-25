"""Local gradient-accumulation wrapper for the official OpenPI trainer.

The upstream training loop calls ``config.optimizer.create(...)`` once and
then applies the returned Optax transformation to the *trainable* parameter
filter on every micro-batch.  Wrapping the complete OpenPI optimizer in
``optax.MultiSteps`` therefore accumulates raw LoRA/action-head gradients,
averages them, and only then runs global-norm clipping and AdamW.

No model or upstream OpenPI source changes are required.  One important
accounting detail remains: OpenPI's ``TrainState.step`` and
``num_train_steps`` count micro-steps, while the wrapped optimizer's schedule
counts emitted optimizer updates.  Save intervals and the final micro-step
count should normally be multiples of ``steps``.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclasses.dataclass(frozen=True)
class GradientAccumulatingOptimizer:
    """Adapt an OpenPI optimizer config to an effective larger batch.

    ``base`` must implement OpenPI's ``OptimizerConfig.create`` protocol.
    ``steps=4`` with OpenPI ``batch_size=1`` gives effective batch size four.
    The complete base transform is nested inside ``MultiSteps`` so gradient
    clipping and Adam moments see the averaged gradient exactly once.
    """

    base: Any
    steps: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or int(self.steps) != self.steps or self.steps < 1:
            raise ValueError("gradient accumulation steps must be a positive integer")
        if not callable(getattr(self.base, "create", None)):
            raise TypeError("base optimizer must provide create(lr, weight_decay_mask=...)")

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: Any | None = None,
    ) -> optax.GradientTransformation:
        base_tx = self.base.create(lr, weight_decay_mask=weight_decay_mask)
        if self.steps == 1:
            return base_tx
        # MultiSteps is a wrapper, NOT an instance of GradientTransformation.
        # OpenPI's runtime-checked TrainState rejects the wrapper even though
        # tiny numerical tests accept its init/update duck type. Export the
        # public transformation; optimizer state/averaging are unchanged.
        return optax.MultiSteps(
            base_tx,
            every_k_schedule=int(self.steps),
            use_grad_mean=True,
        ).gradient_transformation()


def numerical_equivalence_test(steps: int = 4, groups: int = 2) -> dict[str, Any]:
    """Compare accumulated micro-gradients with direct mean-gradient updates.

    This is deliberately a tiny CPU test, independent of model weights.  It
    uses OpenPI's actual AdamW config (including clipping), verifies that all
    pre-boundary updates are exactly zero, and compares two emitted updates so
    Adam moments and learning-rate schedule counters are both exercised.
    Frozen bf16 parameters are represented by a sentinel outside the trainable
    tree, matching the filtering performed by the official train step.
    """

    if steps < 1 or groups < 1:
        raise ValueError("steps and groups must be positive")

    from openpi.training import optimizer as openpi_optimizer

    trainable = {
        "lora_a": jnp.asarray([[0.20, -0.10], [0.05, 0.30]], dtype=jnp.float32),
        "lora_b": jnp.asarray([0.40, -0.25, 0.10], dtype=jnp.float32),
    }
    # The official loop filters frozen parameters before tx.init/tx.update.
    frozen_bf16 = jnp.asarray([1.25, -2.50], dtype=jnp.bfloat16)
    frozen_before = np.asarray(frozen_bf16).copy()

    # A non-constant schedule makes accidental advancement on every
    # micro-step observable in the second emitted update.
    learning_rate = optax.linear_schedule(
        init_value=2.5e-5,
        end_value=5.0e-6,
        transition_steps=max(groups - 1, 1),
    )
    base_config = openpi_optimizer.AdamW(
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=1e-10,
        clip_gradient_norm=1.0,
    )
    accumulated_tx = GradientAccumulatingOptimizer(base_config, steps=steps).create(
        learning_rate
    )
    if not isinstance(accumulated_tx, optax.GradientTransformation):
        raise AssertionError("Accumulating optimizer violates OpenPI TrainState.tx type")
    reference_tx = base_config.create(learning_rate)

    accumulated_params = jax.tree.map(lambda value: value.copy(), trainable)
    reference_params = jax.tree.map(lambda value: value.copy(), trainable)
    accumulated_state = accumulated_tx.init(accumulated_params)
    reference_state = reference_tx.init(reference_params)
    emitted_micro_steps: list[int] = []

    for group in range(groups):
        micro_gradients = []
        for micro in range(steps):
            scale = np.float32(0.17 + 0.11 * group + 0.07 * micro)
            gradients = {
                "lora_a": jnp.asarray(
                    [[scale, -2.0 * scale], [0.5 * scale, 1.5 * scale]],
                    dtype=jnp.float32,
                ),
                "lora_b": jnp.asarray(
                    [1.2 * scale, -0.8 * scale, 0.3 * scale], dtype=jnp.float32
                ),
            }
            micro_gradients.append(gradients)
            before = jax.tree.map(lambda value: value.copy(), accumulated_params)
            updates, accumulated_state = accumulated_tx.update(
                gradients, accumulated_state, accumulated_params
            )
            accumulated_params = optax.apply_updates(accumulated_params, updates)

            global_micro_step = group * steps + micro + 1
            if micro < steps - 1:
                update_norm = float(optax.global_norm(updates))
                if update_norm != 0.0:
                    raise AssertionError(
                        f"optimizer emitted early at micro-step {global_micro_step}: {update_norm}"
                    )
                unchanged = all(
                    np.array_equal(np.asarray(old), np.asarray(new))
                    for old, new in zip(
                        jax.tree_util.tree_leaves(before),
                        jax.tree_util.tree_leaves(accumulated_params),
                        strict=True,
                    )
                )
                if not unchanged:
                    raise AssertionError("parameters changed before accumulation boundary")
            else:
                emitted_micro_steps.append(global_micro_step)

        mean_gradient = jax.tree.map(
            lambda *values: jnp.mean(jnp.stack(values, axis=0), axis=0),
            *micro_gradients,
        )
        reference_updates, reference_state = reference_tx.update(
            mean_gradient, reference_state, reference_params
        )
        reference_params = optax.apply_updates(reference_params, reference_updates)

    errors = [
        float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
        for got, expected in zip(
            jax.tree_util.tree_leaves(accumulated_params),
            jax.tree_util.tree_leaves(reference_params),
            strict=True,
        )
    ]
    max_abs_error = max(errors, default=0.0)
    frozen_unchanged = np.array_equal(frozen_before, np.asarray(frozen_bf16))
    passed = max_abs_error <= 1e-10 and frozen_unchanged
    report = {
        "passed": passed,
        "accumulation_steps": int(steps),
        "optimizer_update_groups": int(groups),
        "micro_steps": int(steps * groups),
        "emitted_micro_steps": emitted_micro_steps,
        "max_parameter_abs_error": max_abs_error,
        "frozen_bf16_unchanged": frozen_unchanged,
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(numerical_equivalence_test(), indent=2))
