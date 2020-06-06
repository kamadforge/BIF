"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'anon_m'

import argparse
import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
# from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
# from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
# from switch_model_wrapper import BinarizedMnistNet
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

from switch_mnist_featimp import load_two_label_mnist_data, hard_select_data, make_select_loader
from mnist_posthoc_accuracy_eval import test_posthoc_acc


"""Instance-wise Variable Selection (INVASE) module - with baseline

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
           "IINVASE: Instance-wise Variable Selection using Neural Networks,"
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com
"""


def invase_select_data(invase_model, loader, device):
  x_data, y_data, selection = [], [], []
  with pt.no_grad():
    for x, y in loader:
      x_data.append(x.numpy())
      y_data.append(y.numpy())
      # x_sel = nnf.softplus(selector(x.to(device)))
      # x_sel = x_sel / pt.sum(x_sel, dim=1)[:, None] * 16  # multiply by patch size
      x_sel, _ = invase_model.select(x.to(device))
      selection.append(x_sel.cpu().numpy())
  return np.concatenate(x_data), np.concatenate(y_data), np.concatenate(selection)




class Net(nn.Module):
  def __init__(self, d_in, d_hid, d_out, act='relu', act_out='logsoftmax', use_bn=True):
    assert act in {'relu', 'selu'}
    assert act_out in {'logsoftmax', 'sigmoid'}
    super(Net, self).__init__()
    self.use_bn = use_bn
    self.act = nn.ReLU() if act == 'relu' else nn.SELU()
    self.act_out = nn.LogSoftmax(dim=1) if act_out == 'logsoftmax' else nn.Sigmoid()
    self.fc1 = nn.Linear(d_in, d_hid)
    self.bn1 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.bn2 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc3 = nn.Linear(d_hid, d_out)

  def forward(self, x_in):
    x = self.fc1(x_in)
    x = self.bn1(x) if self.use_bn else x
    x = self.act(x)
    x = self.fc2(x)
    x = self.bn2(x) if self.use_bn else x
    x = self.act(x)
    x = self.fc3(x)
    x = self.act_out(x)
    return x


class Invase(nn.Module):
  """INVASE class.

  Attributes:
    - x_train: training features
    - y_train: training labels
    - model_type: invase or invase_minus
    - model_parameters:
      - actor_h_dim: hidden state dimensions for actor
      - critic_h_dim: hidden state dimensions for critic
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE
  """

  def __init__(self, model_parameters, device):
    super(Invase, self).__init__()

    self.lamda = model_parameters['lamda']
    self.actor_h_dim = model_parameters['actor_h_dim']
    self.critic_h_dim = model_parameters['critic_h_dim']
    self.batch_size = model_parameters['batch_size']
    self.activation = model_parameters['activation']
    self.learning_rate = model_parameters['learning_rate']
    self.model_type = model_parameters['model_type']

    self.device = device
    self.dim = 784
    self.label_dim = 2


    # Build and compile critic
    self.critic_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                          act_out='logsoftmax').to(self.device)
    # self.critic.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # Build and compile the actor
    self._actor_net = Net(self.dim, self.actor_h_dim, 49, act=self.activation,
                          act_out='sigmoid', use_bn=False).to(self.device)
    # self.actor.compile(loss=self.actor_loss, optimizer=optimizer)

    total_parameters = list(self.critic_net.parameters()) + list(self._actor_net.parameters())

    if self.model_type == 'invase':
      # Build and compile the baseline
      self.baseline_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                              act_out='logsoftmax').to(self.device)
      # self.baseline.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
      total_parameters.extend(list(self.baseline_net.parameters()))

    self.optimizer = pt.optim.Adam(params=total_parameters, lr=self.learning_rate, weight_decay=1e-3)

  def actor_loss(self, selection, log_critic_out, log_baseline_out, y_true, actor_out):
    """Custom loss for the actor.

    Args:
      - y_true:
        - actor_out: actor output after sampling
        - critic_out: critic output
        - baseline_out: baseline output (only for invase)
      - y_pred: output of the actor network

    Returns:
      - loss: actor loss
    """

    selection = selection.detach()
    critic_loss = -pt.sum(y_true * log_critic_out.detach(), dim=1)

    if self.model_type == 'invase':
      # Baseline loss
      baseline_loss = -pt.sum(y_true * log_baseline_out.detach(), dim=1)
      # Reward
      Reward = -(critic_loss - baseline_loss)
    elif self.model_type == 'invase_minus':
      Reward = -critic_loss
    else:
      raise ValueError

    # Policy gradient loss computation.
    actor_term = pt.sum(selection * pt.log(actor_out + 1e-8) + (1 - selection) * pt.log(1 - actor_out + 1e-8), dim=1)
    sparcity_term = pt.mean(actor_out, dim=1)
    custom_actor_loss = Reward * actor_term - self.lamda * sparcity_term

    # custom actor loss
    custom_actor_loss = pt.mean(-custom_actor_loss)

    return custom_actor_loss

  def importance_score(self, x):
    """Return feature importance score.

    Args:
      - x: feature

    Returns:
      - feature_importance: instance-wise feature importance for x
    """
    return_numpy = False
    if not isinstance(x, pt.Tensor):
      x = pt.tensor(x, device=self.device, dtype=pt.float32)
      return_numpy = True

    _, feature_importance = self.select(x)
    if return_numpy:
      feature_importance = feature_importance.cpu().detach().numpy()
    return feature_importance

  def predict(self, x):
    """
    Predict outcomes.
    Args:
      - x: feature

    Returns:
      - y_hat: predictions
    """
    return_numpy = False
    if not isinstance(x, pt.Tensor):
      x = pt.tensor(x, device=self.device, dtype=pt.float32)
      return_numpy = True
    # Generate a batch of selection probability
    selection, _ = self.select(x)
    # Sampling the features based on the selection_probability
    # selection = pt.bernoulli(selection_probability)
    # Prediction
    y_hat = pt.exp(self.critic_net(x * selection))
    if return_numpy:
      y_hat = y_hat.cpu().detach().numpy()
    return y_hat

  def select(self, x):
    low_res_actor_out = self._actor_net(x)
    low_res_selection = pt.bernoulli(low_res_actor_out)
    # upscale to patches, first output, then sampled selection
    low_res_actor_out = low_res_actor_out.view(-1, 7, 7)
    high_res_actor_out = pt.repeat_interleave(pt.repeat_interleave(low_res_actor_out, 4, dim=1), 4, dim=2)
    high_res_actor_out = high_res_actor_out.reshape(-1, 784)

    low_res_selection = low_res_selection.view(-1, 7, 7)
    high_res_selection = pt.repeat_interleave(pt.repeat_interleave(low_res_selection, 4, dim=1), 4, dim=2)
    high_res_selection = high_res_selection.reshape(-1, 784)
    # Sampling the features based on the selection_probability
    return high_res_selection, high_res_actor_out


def train_model(model, learning_rate, n_epochs, train_loader, test_loader, device):
  adam = pt.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)

  # filepath = f"models/mnist/model.pt"

  for ep in range(n_epochs):

    model.train(True)
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)

      y_batch_onehot = pt.stack([1-y_batch, y_batch], dim=1)

      adam.zero_grad()
      selection, actor_out = model.select(x_batch)
      log_critic_out = model.critic_net(x_batch * selection)
      critic_loss = nnf.nll_loss(log_critic_out, y_batch)
      combined_loss = critic_loss

      if model.model_type == 'invase':
        log_baseline_out = model.baseline_net(x_batch)
        # print(pt.max(log_baseline_out), pt.min(log_baseline_out))
        # print(pt.max(y_batch), pt.min(y_batch))
        baseline_loss = nnf.nll_loss(log_baseline_out, y_batch)
        combined_loss += baseline_loss
      elif model.model_type == 'invase_minus':
        log_baseline_out = None
      else:
        raise ValueError

      actor_loss = model.actor_loss(selection, log_critic_out, log_baseline_out, y_batch_onehot, actor_out)
      combined_loss += actor_loss

      combined_loss.backward()
      adam.step()

    model.eval()

    summed_loss = 0
    correct_preds = 0
    n_tested = 0
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1)

      test_selection, _ = model.select(x_batch)
      critic_pred = model.critic_net(x_batch * test_selection)
      loss = nnf.nll_loss(critic_pred, y_batch)

      summed_loss += loss.item() * y_batch.shape[0]
      matches = pt.max(critic_pred, dim=1)[1] == y_batch
      correct_preds += pt.sum(matches).item()
      n_tested += y_batch.shape[0]

    print(f'epoch {ep} done. Acc: {correct_preds / n_tested}, Loss: {summed_loss / n_tested}')


def test_classifier_epoch(classifier, test_loader, device):
  classifier.eval()
  test_loss = 0
  correct = 0
  with pt.no_grad():
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      pred = classifier(x_batch)
      test_loss = nnf.binary_cross_entropy_with_logits(pred, y_batch, reduction='sum').item()
      # test_loss = nnf.nll_loss(pred, y_batch, reduction='sum').item()  # sum up batch loss
      class_pred = pt.sigmoid(pred).round()  # get the index of the max log-probability
      # print(class_pred.shape, y_batch.shape)
      correct += class_pred.eq(y_batch.view_as(class_pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def train_classifier(classifier, train_loader, test_loader, epochs, lr, device):
  optimizer = optim.Adam(classifier.parameters(), lr=lr)

  for epoch in range(epochs):  # loop over the dataset multiple times
    classifier.train()
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()

      loss = nnf.binary_cross_entropy_with_logits(classifier(x_batch), y_batch)
      # loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg)
      loss.backward()
      optimizer.step()

    test_classifier_epoch(classifier, test_loader, device)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, default='mnist')
  # parser.add_argument('--selected-label', type=int, default=3)  # label for 1-v-rest training
  # parser.add_argument('--log-interval', type=int, default=500)
  # parser.add_argument('--n-switch-samples', type=int, default=3)

  parser.add_argument('--actor_h_dim', help='hidden state dimensions for actor', default=100, type=int)
  parser.add_argument('--critic_h_dim', help='hidden state dimensions for critic', default=200, type=int)
  parser.add_argument('--activation', help='activation function of the networks',
                      choices=['selu', 'relu'], default='relu', type=str)
  parser.add_argument('--learning_rate', help='learning rate of model training', default=0.0001, type=float)
  parser.add_argument('--model_type', help='inavse or invase- (without baseline)',
                      choices=['invase', 'invase_minus'], default='invase', type=str)

  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  #  lamda = 15.5 --> 5 feats
  #  lamda = 18.5 --> 4 feats
  #  lamda = 22.5 --> 3 feats
  #  lamda = 105.5 --> 1 feats
  parser.add_argument('--lamda', help='inavse hyper-parameter lambda', default=100., type=float)
  parser.add_argument('--seed', type=int, default=6)

  return parser.parse_args()


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  # train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  train_loader, test_loader = load_two_label_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size,
                                                        data_path='../../data',
                                                        label_a=ar.label_a, label_b=ar.label_b,
                                                        tgt_type=np.int64)

  model_parameters = {'lamda': ar.lamda,
                      'actor_h_dim': ar.actor_h_dim,
                      'critic_h_dim': ar.critic_h_dim,
                      'batch_size': ar.batch_size,
                      'activation': ar.activation,
                      'learning_rate': ar.learning_rate,
                      'model_type': ar.model_type}

  model = Invase(model_parameters, device)
  # d_in=784, d_out=1, datatype=None, n_key_features=ar.select_k, device=device).to(device)
  train_model(model, ar.lr, ar.epochs, train_loader, test_loader, device)
  # , ar.point_estimate, ar.n_switch_samples, ar.alpha_0, n_features, n_data, ar.KL_reg)

  print('Finished Training Selector')
  x_ts, y_ts, ts_selection = invase_select_data(model, test_loader, device)
  # y_ts = y_ts.astype(np.int64)
  x_ts_select = x_ts * ts_selection
  print('average number of selected patches: ', np.mean(np.sum(ts_selection, axis=1))/16)
  # x_ts_select = hard_select_data(x_ts, ts_selection, k=ar.select_k)
  select_test_loader = make_select_loader(x_ts_select, y_ts, train=False, batch_size=ar.test_batch_size,
                                          use_cuda=use_cuda, data_path='../../data')
  print('testing classifier')

  test_posthoc_acc(ar.label_a, ar.label_b, select_test_loader, device, model_path_prefix='../../code/')


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  do_featimp_exp(ar)


if __name__ == '__main__':
  # main()

  # K=5, S=2: 92.3
  # K=5, S=3: 92.4
  # K=5, S10: 85.6
  # K=5, S13: 91.9
  # K=5, S15: 90.3
  # K=4, S12: 91.7  15.5
  # K=4, S=1: 90.8
  # K=4, S=8: 91.6
  # K=4, S=4: 91.7
  # K=4, S10: 91.7
  # K=3, S14: 91.5 15.5
  # K=3, S=2: 89.1 18.5
  # K=3, S=1: 91.7 23
  # K=3, S=2: 89.2 23
  # K=3, S=3: 89.2 23
  # K=2, S=2: 76.3 50
  # K=2, S=3: 76.3 50
  # K=2, S=4: 76.3 50
  # K=2, S=7: 84.7 50
  # K=2, S=8: 76.4 50
  # K=1, S=1: 55.3 100
  # K=1, S=3: 55.3 100
  # K=1, S=4: 75.2
  # K=1, S=5: 55.5
  # K=1, S=6: 50.9

  k1_res = [92.3, 92.4, 85.6, 91.9, 90.3]
  k2_res = [91.7, 90.8, 91.6, 91.7, 91.7]
  k3_res = [91.5, 89.1, 91.7, 89.2, 89.2]
  k4_res = [76.3, 76.3, 76.3, 84.7, 76.4]
  k5_res = [55.3, 55.3, 75.2, 55.5, 50.9]

  print('k1_avg =', sum(k1_res) / 5)
  print('k2_avg =', sum(k2_res) / 5)
  print('k3_avg =', sum(k3_res) / 5)
  print('k4_avg =', sum(k4_res) / 5)
  print('k5_avg =', sum(k5_res) / 5)