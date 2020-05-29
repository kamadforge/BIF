"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung&frederik'

import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn.parameter import Parameter
from torch.distributions import Gamma


def point_est(phi, x, classifier):
  phi_sum = pt.sum(phi, dim=1)
  # print(x.shape, phi.shape, phi_sum.shape)
  x_select = x * (phi / phi_sum[:, None])
  return classifier(x_select)  # just batch size


def sample_est(phi, x, classifier, n_samples):
  concentration_param = phi.view(-1, 1).repeat(1, n_samples)
  beta_param = pt.ones(concentration_param.size())
  # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
  gamma_samps = Gamma(concentration_param, beta_param).rsample()  # 200, 150, input_dim x samples_num
  assert not any(pt.sum(gamma_samps, 0) == 0)

  sample_stack = gamma_samps / pt.sum(gamma_samps, 0)  # input dim by  # samples

  x_samps = pt.einsum("ij,jk -> ikj", (x, sample_stack))
  bs, n_samp, n_feat = x_samps.shape
  x_samps = x_samps.reshape(bs * n_samp, n_feat)
  model_out = classifier(x_samps)
  return model_out.view(bs, n_samp)


class SwitchWrapper(nn.Module):

  def __init__(self, trained_model, input_dim, num_samps_for_switch, point_estimate):
    # def __init__(self, input_dim, hidden_dim):
    super(SwitchWrapper, self).__init__()

    self.trained_model = trained_model
    self.parameter = Parameter(-1e-10*pt.ones(input_dim), requires_grad=True)
    self.num_samps_for_switch = num_samps_for_switch
    self.point_estimate = point_estimate

  def forward(self, x):  # x is mini_batch_size by input_dim

    phi = nnf.softplus(self.parameter)

    if any(pt.isnan(phi)):
      print("some Phis are NaN")
    # it looks like too large values are making softplus-transformed values very large and returns NaN.
    # this occurs when optimizing with a large step size (or/and with a high momentum value)

    if self.point_estimate:
      model_out = point_est(phi, x, self.classifier)

    else:
      model_out = sample_est(phi, x, self.classifier, self.num_samps_for_switch)

    return model_out, phi


# def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
def loss_function(prediction, true_y, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg):
  print(prediction.shape, true_y.shape)
  BCE = nnf.binary_cross_entropy(prediction, true_y, reduction='mean')

  if KL_reg:
    # KLD term
    alpha_0 = pt.tensor(alpha_0, dtype=pt.float32)
    hidden_dim = pt.tensor(n_features, dtype=pt.float32)
    trm1 = pt.lgamma(pt.sum(phi_cand)) - pt.lgamma(hidden_dim*alpha_0)
    trm2 = - pt.sum(pt.lgamma(phi_cand)) + hidden_dim*pt.lgamma(alpha_0)
    trm3 = pt.sum((phi_cand-alpha_0)*(pt.digamma(phi_cand)-pt.digamma(pt.sum(phi_cand))))

    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    return BCE + annealing_rate * KLD / n_data

  else:

    return BCE


class SwitchNetWrapper(nn.Module):

  def __init__(self, selector, classifier, input_dim, num_samps_for_switch, do_point_estimate):
    # def __init__(self, input_dim, hidden_dim):
    super(SwitchNetWrapper, self).__init__()

    self.selector = selector
    self.classifier = classifier

    self.num_samps_for_switch = num_samps_for_switch
    self.do_point_estimate = do_point_estimate

  def forward(self, x):  # x is mini_batch_size by input_dim

    phi = nnf.softplus(self.selector(x))
    # assert not any(pt.isnan(phi))

    if self.do_point_estimate:
      model_out = point_est(phi, x, self.classifier)
    else:
      model_out = sample_est(phi, x, self.classifier, self.num_samps_for_switch)
    return model_out, phi

  def selector_params(self):
    return self.selector.parameters()


class MnistPatchSelector(nn.Module):
  def __init__(self, d_hid=300):
    super(MnistPatchSelector, self).__init__()
    self.fc1 = nn.Linear(784, d_hid)
    self.fc2 = nn.Linear(d_hid, 49)

  def forward(self, x):
    x = pt.flatten(x, 1)
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.fc2(x)

    x = x.view(-1, 7, 7)
    x = pt.repeat_interleave(pt.repeat_interleave(x, 4, dim=1), 4, dim=2)

    return x.view(-1, 784)


class MnistGlobalPatches(nn.Module):
  def __init__(self, d_hid=300):
    super(MnistGlobalPatches, self).__init__()
    self.param = nn.Parameter(pt.randn(1, 7, 7), requires_grad=True)

  def forward(self, x):
    param = self.param.expand(x.shape[0], 7, 7)
    param = pt.repeat_interleave(pt.repeat_interleave(param, 4, dim=1), 4, dim=2)
    return param.view(-1, 784)




class LogReg(nn.Module):
  def __init__(self, d_in, d_out):
    super(LogReg, self).__init__()
    self.fc = nn.Linear(d_in, d_out, bias=False)

  def forward(self, x_in):
    return pt.sigmoid(self.fc(x_in))

  def load(self, weight_mat):
    self.fc.weight.data = weight_mat


class MnistNet(nn.Module):
  def __init__(self, selected_label=None):
    super(MnistNet, self).__init__()
    self.selected_label = selected_label
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = pt.flatten(x, 1)
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.fc2(x)

    if self.selected_label is None:
      output = nnf.log_softmax(x, dim=1)
    else:
      # for selected labels, the output of this network changed to values after the softmax operation
      output = nnf.softmax(x, dim=1)[:, self.selected_label]

    return output


class BinarizedMnistNet(nn.Module):
  def __init__(self, d_hid=300):
    super(BinarizedMnistNet, self).__init__()

    self.fc1 = nn.Linear(784, d_hid)
    self.fc2 = nn.Linear(d_hid, 1)

  def forward(self, x):
    x = pt.flatten(x, 1)
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.fc2(x)
    x = x.flatten()
    return x  # assume BCE with logits as loss
