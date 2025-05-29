from typing import Callable, Optional

import jax
import optax


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
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(xy_arg, params_update)

    if have_aux:
      agent0_loss, agent0_metrics = agent0_result
      agent1_loss, agent1_metrics = agent1_result
      result = (agent0_loss + agent1_loss, agent0_metrics | agent1_metrics)
    else:
      result = (agent0_result + agent1_result)

    return result, params, optimizer_state

  return f


def cgd_update_fn(
    agent0_linear_obj_term,
    agent1_linear_obj_term,
    bilinear_obj_term,
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
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

  agent0_linear_obj_and_pgrad_fn = loss_and_pgrad(
    agent0_linear_obj_term, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )
  agent1_linear_obj_and_pgrad_fn = loss_and_pgrad(
    agent1_linear_obj_term, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )
  bilinear_obj_and_pgrad_fn = loss_and_pgrad(
    bilinear_obj_term, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )

  def f(*args, optimizer_state):
    breakpoint()
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state

  return f
