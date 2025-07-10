from typing import Callable, Tuple, Optional, Sequence, NamedTuple

import jax
import jax.numpy as jnp
import optax
import tree_math as tm
import functools

from jax.tree_util import tree_map, tree_structure, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from .utils import conjugate_gradient, cross_hvp


def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def gda_update_fn(
    agent0_linear_loss_term,
    agent1_linear_loss_term,
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    have_aux: bool = False,
):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """

  agent0_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent0_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )
  agent1_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent1_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )

  def f(*args, optimizer_state):
    xy_arg, rest = args[0], args[1:]
    agent0_result, agent0_grads = agent0_linear_loss_and_pgrad_fn(xy_arg[0], *rest)
    agent1_result, agent1_grads = agent1_linear_loss_and_pgrad_fn(xy_arg[1], *rest)
    grads = [agent0_grads, jax.tree.map(lambda x: -x, agent1_grads)]
    params_update, optimizer_state = optimizer.update(grads, optimizer_state, params=xy_arg)
    params = optax.apply_updates(xy_arg, params_update)

    if have_aux:
      agent0_loss, agent0_metrics = agent0_result
      agent1_loss, agent1_metrics = agent1_result
      result = (agent0_loss + agent1_loss, agent0_metrics | agent1_metrics)
    else:
      result = (agent0_result + agent1_result)

    return result, params, optimizer_state

  return f


def eg_update_fn(
    agent0_linear_loss_term,
    agent1_linear_loss_term,
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    have_aux: bool = False,
):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """

  agent0_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent0_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )
  agent1_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent1_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )

  def f(*args, optimizer_state):
    xy_arg, rest = args[0], args[1:]
    agent0_result, agent0_grads = agent0_linear_loss_and_pgrad_fn(xy_arg[0], *rest)
    agent1_result, agent1_grads = agent1_linear_loss_and_pgrad_fn(xy_arg[1], *rest)
    grads = [agent0_grads, jax.tree.map(lambda x: -x, agent1_grads)]
    params_update_0, optimizer_state = optimizer.update(grads, optimizer_state, params=xy_arg)
    xy_arg_onehalf = optax.apply_updates(xy_arg, params_update_0)

    _, agent0_grads_onehalf = agent0_linear_loss_and_pgrad_fn(xy_arg_onehalf[0], *rest)
    _, agent1_grads_onehalf = agent1_linear_loss_and_pgrad_fn(xy_arg_onehalf[1], *rest)
    grads_onehalf = [agent0_grads_onehalf, jax.tree.map(lambda x: -x, agent1_grads_onehalf)]
    params_update_1, optimizer_state = optimizer.update(grads_onehalf, optimizer_state, params=xy_arg)
    params = optax.apply_updates(xy_arg, params_update_1)

    if have_aux:
      agent0_loss, agent0_metrics = agent0_result
      agent1_loss, agent1_metrics = agent1_result
      result = (agent0_loss + agent1_loss, agent0_metrics | agent1_metrics)
    else:
      result = (agent0_result + agent1_result)

    return result, params, optimizer_state

  return f


class CGOState(NamedTuple):
  old_x: jnp.ndarray
  old_y: jnp.ndarray
  solve_x: bool


CGDState = CGOState


def cgo_init_state(params):
  x, y = params
  return CGDState(old_x=x, old_y=y, solve_x=True)


cgd_init_state = cgo_init_state


def solve_for_cgo_update(
    f: Callable[[any, any], jnp.ndarray],
    x: any,
    y: any,
    grad_x: any,
    grad_y: any,
    hessian_xy_mult_grad_y: any,
    alpha: float,
    nsteps: int = 100,
    init: any = None,
    for_x: bool = True,
    residual_tol: float = 1e-12,
) -> any:
    """
    Solves the linear system:
        (I + alpha^2 D_{xy} f D_{yx} f) x_delta = -D_x f - alpha D_{xy} f D_y f
    and returns x_delta as a pytree.
    """

    # compute RHS b = -grad_x - eta * D_xy grad_y
    b = jax.tree.map(lambda a, b: -a - alpha * b, grad_x, hessian_xy_mult_grad_y)
    b_vec = tm.Vector(b)

    # define matvec: v -> (I + alpha^2 D_{xy} D_{yx}) v
    def matvec(v):
        hvp_1, _ = cross_hvp(f, x, y, v.tree, order="yx")
        hvp_2, _ = cross_hvp(f, x, y, hvp_1, order="xy")
        return v + alpha**2 * tm.Vector(hvp_2)

    # initial CG state
    xk = tm.Vector(jax.tree.map(jnp.zeros_like, b)) if init is None else tm.Vector(init)
    rk = tm.Vector(b)
    pk = rk
    rdotr = rk.dot(rk)
    rdotr_init = rdotr

    def cond_fn(state):
        _, rk, _, rdotr = state
        return rdotr > residual_tol * rdotr_init

    def body_fn(state):
        xk, rk, pk, rdotr = state
        Apk = matvec(pk)
        alpha = rdotr / pk.dot(Apk)
        xk_new = xk + alpha * pk
        rk_new = rk - alpha * Apk
        new_rdotr = rk_new.dot(rk_new)
        beta = new_rdotr / rdotr
        pk_new = rk_new + beta * pk
        return xk_new, rk_new, pk_new, new_rdotr

    # Run the CG loop with early stopping
    init_state = (xk, rk, pk, rdotr)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    xk_final, *_ = final_state
    return xk_final.tree


def cgo_update_fn(
    agent0_linear_loss_term,
    agent1_linear_loss_term,
    bilinear_loss_term,
    optimizer: optax.GradientTransformation,
    eta: float,
    pmap_axis_name: Optional[str],
    alpha: Optional[float] = None,
    have_aux: bool = False,
):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    have_aux: Whether the loss_fns have auxiliary data.
    eta: learning rate
    alpha: bilinear term sensitivity term

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """

  agent0_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent0_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )
  agent1_linear_loss_and_pgrad_fn = loss_and_pgrad(
    agent1_linear_loss_term, pmap_axis_name=pmap_axis_name, has_aux=have_aux
  )

  if alpha is None:
    alpha = eta

  def f(*args, optimizer_state, cgo_state):
    xy_arg, rest = args[0], args[1:]
    x, y = xy_arg
    x_result, grad_x = agent0_linear_loss_and_pgrad_fn(x, *rest)
    y_result, grad_y = agent1_linear_loss_and_pgrad_fn(y, *rest)
    bilinear_result = bilinear_loss_term(xy_arg, *rest)

    def diadic_bilinear_loss_term(x, y):
      return bilinear_loss_term([x, y], *rest)

    # D_xy mult D_y
    hvp_x, bi_grad_y = cross_hvp(
      diadic_bilinear_loss_term,
      x, y, grad_y,
      order = "xy",
      has_aux=have_aux,
    )

    # D_yx mult D_x
    hvp_y, bi_grad_x = cross_hvp(
      diadic_bilinear_loss_term,
      x, y, grad_x,
      order = "yx",
      has_aux=have_aux,
    )

    def solve_for_x(cgo_state):
      delta_x = solve_for_cgo_update(diadic_bilinear_loss_term, x, y, grad_x, grad_y, hvp_x,
                                     alpha, for_x=True)
      d_yx_mult_delta_x, _ = cross_hvp(diadic_bilinear_loss_term, x, y, delta_x)
      delta_y = tm.Vector(grad_y) + eta * tm.Vector(d_yx_mult_delta_x)
      return [delta_x, delta_y.tree], cgo_state._replace(old_y=delta_y.tree)

    def solve_for_y(cgo_state):
      delta_y = solve_for_cgo_update(diadic_bilinear_loss_term, y, x, grad_y, grad_x, hvp_y,
                                     alpha, for_x=False)
      d_xy_mult_delta_y, _ = cross_hvp(diadic_bilinear_loss_term, x, y, delta_y)
      delta_x = tm.Vector(grad_x) - eta * tm.Vector(d_xy_mult_delta_y)
      return [delta_x.tree, delta_y], cgo_state._replace(old_x=delta_x.tree)

    delta_xy, cgo_state = jax.lax.cond(cgo_state.solve_x, solve_for_x, solve_for_y, cgo_state)
    cgo_state = cgo_state._replace(solve_x=~cgo_state.solve_x)

    params_update, optimizer_state = optimizer.update(delta_xy, optimizer_state, params=xy_arg)
    params = optax.apply_updates(xy_arg, params_update)

    if have_aux:
      x_loss, x_metrics = x_result
      y_loss, y_metrics = y_result
      result = (x_loss + y_loss, x_metrics | y_metrics)
    else:
      result = (x_result + y_result)

    return result, params, optimizer_state

  return f


cgd_update_fn = functools.partial(cgo_update_fn, alpha=None)
