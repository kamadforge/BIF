import torch as pt
from backpack import backpack
from backpack.extensions import BatchGrad, BatchL2Grad


def dp_sgd_backward(params, loss, device, clip_norm, noise_factor):
  """
  the models containing params must have been "extended" after initialization by calling
  model = backpack.extend(model)
  other than that, this function can simply replace the normal model.backward call without further changes
  :param params: parameters to train
  :param loss: computed loss. Must allow sample-wise gradients
  :param device: cpu/gpu key on which the model is run
  :param clip_norm:
  :param noise_factor:
  :return:
  """
  if not isinstance(params, list):
    params = [p for p in params]

  with backpack(BatchGrad(), BatchL2Grad()):
    loss.backward()

  squared_param_norms = [p.batch_l2 for p in params]  # first we get all the squared parameter norms...
  global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))  # ...then compute the global norms...
  global_clips = pt.clamp_max(clip_norm / global_norms, 1.)  # ...and finally get a vector of clipping factors

  for idx, param in enumerate(params):
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips, param.grad_batch)
    clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch

    noise_sdev = noise_factor * 2 * clip_norm  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
    perturbed_grad = clipped_grad + pt.randn_like(clipped_grad, device=device) * noise_sdev  # ...and applied
    param.grad = perturbed_grad  # now we set the parameter gradient to what we just computed

  return global_norms, global_clips


def expand_vector(vec, tgt_tensor):
  tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)