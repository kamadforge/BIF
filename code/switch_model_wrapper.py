"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma


class SwitchWrapper(nn.Module):

  def __init__(self, trained_model, input_dim, num_samps_for_switch):
    # def __init__(self, input_dim, hidden_dim):
    super(SwitchWrapper, self).__init__()

    self.trained_model = trained_model
    self.parameter = Parameter(-1e-10*torch.ones(input_dim), requires_grad=True)
    self.num_samps_for_switch = num_samps_for_switch

  def forward(self, x):  # x is mini_batch_size by input_dim

    phi = F.softplus(self.parameter)

    if any(torch.isnan(phi)):
      print("some Phis are NaN")
    # it looks like too large values are making softplus-transformed values very large and returns NaN.
    # this occurs when optimizing with a large step size (or/and with a high momentum value)

    """ draw Gamma RVs using phi and 1 """
    num_samps = self.num_samps_for_switch
    concentration_param = phi.view(-1, 1).repeat(1, num_samps)
    beta_param = torch.ones(concentration_param.size())
    # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
    Gamma_obj = Gamma(concentration_param, beta_param)
    gamma_samps = Gamma_obj.rsample()  # 200, 150, input_dim x samples_num

    if any(torch.sum(gamma_samps, 0) == 0):
      print("sum of gamma samps are zero!")
    else:
      Sstack = gamma_samps / torch.sum(gamma_samps, 0)  # input dim by  # samples

    SstackT = Sstack.t()
    # x_samps = torch.einsum("ij,jk -> ijk", (x, Sstack))

    output = torch.einsum('ij, mj -> imj', (SstackT, x))  # samples, batchsize, dimension
    output = output.reshape(output.shape[0] * output.shape[1], output.shape[2]) # samples*batchsize, dimension

    # x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W)))
    # labelstack = torch.sigmoid(x_out)
    # x_samps = torch.einsum("ij,jk -> ikj", (x, Sstack))
    # bs, n_samp, n_feat = x_samps.shape
    # # print(bs, n_samp, n_feat)
    # x_samps = x_samps.reshape(bs * n_samp, n_feat)

    # print(x_samps.shape)
    # model_out = self.trained_model(x_samps)

    output = self.trained_model(output)

    output = output.reshape(self.num_samps_for_switch, x.shape[0])
    output = output.transpose(0, 1)

    # model_out = model_out.view(bs, n_samp)
    # model_out = torch.transpose(model_out, 1, 2)
    return output, phi


# def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
def loss_function(prediction, true_y, phi_cand, alpha_0, n_features, n_data, annealing_rate):

  BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

  # this was for sanity check
  # BCE_mat = torch.zeros(prediction.shape[1])
  # for ind in torch.arange(0, prediction.shape[1]):
  #   BCE_mat[ind] = F.binary_cross_entropy(prediction[:,ind], true_y[:,ind])
  #
  # BCE = torch.mean(BCE_mat)


  # loss = nn.CrossEntropyLoss()
  #
  # BCE_mat = torch.zeros(prediction.shape[1])
  # for ind in torch.arange(0, prediction.shape[1]):
  #   y_pred = prediction[:, ind, :]
  #   BCE_mat[ind] = loss(y_pred, true_y)
  #
  # BCE = torch.mean(BCE_mat)

  # # KLD term
  # alpha_0 = torch.tensor(alpha_0, dtype=torch.float32)
  # hidden_dim = torch.tensor(n_features, dtype=torch.float32)
  #
  # trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
  # trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
  # trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))
  #
  # KLD = trm1 + trm2 + trm3
  # # annealing kl-divergence term is better

  # return BCE + annealing_rate * KLD / n_data
  return BCE



class LogReg(nn.Module):
  def __init__(self, d_in, d_out):
    super(LogReg, self).__init__()
    self.fc = nn.Linear(d_in, d_out, bias=False)

  def forward(self, x_in):
    return torch.sigmoid(self.fc(x_in))

  def load(self, weight_mat):
    self.fc.weight.data = weight_mat


class MnistNet(nn.Module):
  def __init__(self, selected_label=None):
    super(MnistNet, self).__init__()
    self.selected_label = selected_label
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    if self.selected_label is None:
      output = F.log_softmax(x, dim=1)
    else:
      output = F.softmax(x, dim=1)[:, self.selected_label]

    return output
