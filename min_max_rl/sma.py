import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Any


class SMAState(NamedTuple):
    buffer: Any   # Buffer of [H, ...] for each parameter
    idx: int
    count: int


def simple_moving_average(H):
    def init_fn(params):
        # Initialize buffer with H copies of params
        buffer = jax.tree.map(lambda p: jnp.stack([jnp.zeros_like(p)] * H), params)
        return SMAState(buffer, 0, 0)

    def update_fn(updates, state, params):
        # Insert current params into buffer at idx
        def update_buffer(buf, p):
            buf = buf.at[state.idx].set(p)
            return buf

        buffer = jax.tree.map(update_buffer, state.buffer, params)
        new_count = jnp.minimum(state.count + 1, H)
        new_idx = (state.idx + 1) % H

        # Compute average over buffer
        def avg(buf):
            return buf.sum(axis=0) / new_count
        avg_params = jax.tree.map(avg, buffer)

        # The *additional* update to apply: avg_params - params + updates
        chained_update = jax.tree.map(
            lambda avg_p, p, upd: (avg_p - p) + upd, avg_params, params, updates
        )

        new_state = SMAState(buffer, new_idx, new_count)
        return chained_update, new_state

    return optax.GradientTransformation(init_fn, update_fn)
