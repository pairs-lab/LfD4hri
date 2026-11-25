"""Run in OpenPI venv with JAX_PLATFORMS=cpu; no large checkpoint/GPU."""
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models.model import BaseModel
from openpi.training.optimizer import AdamW
from openpi.training.utils import TrainState
from VLA.pi0.accumulation import GradientAccumulatingOptimizer, numerical_equivalence_test


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__(action_dim=7, action_horizon=1, max_token_len=1)
        self.weight = nnx.Param(jnp.asarray([1.], dtype=jnp.float32))

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.weight.value.reshape(1, 1)**2

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.zeros((1, 1, 7))


class NativeAccumulationTest(unittest.TestCase):
    def test_native_trainstate_constructor_and_jit(self):
        graph, params = nnx.split(TinyModel())
        tx = GradientAccumulatingOptimizer(AdamW(), steps=4).create(1e-3)
        state = TrainState(step=0, params=params, model_def=graph,
                           opt_state=tx.init(params), tx=tx, ema_decay=None)

        @jax.jit
        def microstep(state):
            grad = jax.tree.map(jnp.ones_like, state.params)
            updates, opt_state = state.tx.update(grad, state.opt_state, state.params)
            return state.replace(step=state.step+1, params=optax.apply_updates(state.params, updates),
                                 opt_state=opt_state)

        before = state.params["weight"].value.copy()
        for number in range(4):
            state = microstep(state)
            if number < 3:
                self.assertTrue(jnp.array_equal(before, state.params["weight"].value))
        self.assertFalse(jnp.array_equal(before, state.params["weight"].value))
        self.assertEqual(int(state.step), 4)
        self.assertEqual(int(state.opt_state.gradient_step), 1)

    def test_legacy_wrapper_reproduces_type_error(self):
        graph, params = nnx.split(TinyModel())
        legacy = optax.MultiSteps(AdamW().create(1e-3), every_k_schedule=4)
        self.assertNotIsInstance(legacy, optax.GradientTransformation)
        from jaxtyping import TypeCheckError
        with self.assertRaises(TypeCheckError):
            TrainState(step=0, params=params, model_def=graph,
                       opt_state=legacy.init(params), tx=legacy, ema_decay=None)

    def test_mean_gradients_and_schedule_equivalence(self):
        self.assertTrue(numerical_equivalence_test()["passed"])


if __name__ == "__main__":
    unittest.main()
