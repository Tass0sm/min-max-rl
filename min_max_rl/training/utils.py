from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
from jax.flatten_util import ravel_pytree


def cross_hvp(
    f: Callable[[any, any], jnp.ndarray],
    x: any,
    y: any,
    v: any,
    order = "xy",
    has_aux: bool = True,
):
    """
    Returns  H_(xy or yx) @ v without forming H.
    - H_xy is the quadrant of the hessian where the elements are d / dx_j d / dy_i f
    - x, y may be arbitrary pytrees (dense params, dicts, lists, …).
    - v must have exactly the same pytree structure / dtypes / shapes as x or y.
    """

    if has_aux:
        # keep only the scalar output for differentiation
        def scalar_f(*args, **kwargs):
          return f(*args, **kwargs)[0]
    else:
        def scalar_f(*args, **kwargs):
          return f(*args, **kwargs)

    if order == "xy":
        # define the gradient with respect to the second parameter
        _g = jax.grad(scalar_f, argnums=1)

        # define a function of only the first parameter for the gradient
        def g(_x):
            return _g(_x, y)

        # take the jacobian vector product of that function at (x,) to get the
        # hessian_xy at x,y multiplied by the vector (v,). Return the second
        # return value (tangents, the product result) and the gradient with
        # respect to y at xy.
        return jax.jvp(g, (x,), (v,))[1], g(x)
    elif order == "yx":
        _g = jax.grad(scalar_f, argnums=0)
        def g(_y):
            return _g(x, _y)
        return jax.jvp(g, (y,), (v,))[1], g(y)
    else:
        raise NotImplementedError()
