"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

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
from switch_model_wrapper import BinarizedMnistNet
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

from switch_mnist_featimp import load_two_label_mnist_data, hard_select_data, make_select_loader
from mnist_posthoc_accuracy_eval import test_posthoc_acc

def l2x_select_data(l2x_model, loader, device):
  x_data, y_data, selection = [], [], []
  with pt.no_grad():
    for x, y in loader:
      x_data.append(x.numpy())
      y_data.append(y.numpy())
      # x_sel = nnf.softplus(selector(x.to(device)))
      # x_sel = x_sel / pt.sum(x_sel, dim=1)[:, None] * 16  # multiply by patch size
      _, x_sel = l2x_model.get_selection(x.to(device))
      selection.append(x_sel.cpu().numpy())
  return np.concatenate(x_data), np.concatenate(y_data), np.concatenate(selection)


class L2XModel(nn.Module):
  def __init__(self, d_in, d_out, datatype, n_key_features, device, d_hid_sel=100, d_hid_clf=200, tau=0.1):
    super(L2XModel, self).__init__()

    self.act = nn.ReLU() if datatype in ['orange_skin', 'XOR'] else nn.SELU()
    self.d_out = d_out

    # q(S|X)
    self.fc1 = nn.Linear(d_in, d_hid_sel)
    self.fc2 = nn.Linear(d_hid_sel, d_hid_sel)
    self.fc3 = nn.Linear(d_hid_sel, 49)

    # concrete sampling
    self.tau = tau
    self.n_key_features = n_key_features
    self.device = device

    # q(X_S)
    self.fc4 = nn.Linear(d_in, d_hid_clf)
    self.bn4 = nn.BatchNorm1d(d_hid_clf)
    self.fc5 = nn.Linear(d_hid_clf, d_hid_clf)
    self.bn5 = nn.BatchNorm1d(d_hid_clf)
    self.fc6 = nn.Linear(d_hid_clf, d_out)

  def forward(self, x_in):
    x_select, _ = self.get_selection(x_in)
    pred = self.classify_selection(x_select)

    if self.d_out == 1:
      pred = pt.sigmoid(pred.view(-1))
    return pred

  def get_selection(self, x_in):
    x = self.fc1(x_in)
    x = self.act(x)
    x = self.fc2(x)
    x = self.act(x)
    x = self.fc3(x)

    low_res_selection = self._sample_concrete(x)
    low_res_selection = low_res_selection.view(-1, 7, 7)
    high_res_selection = pt.repeat_interleave(pt.repeat_interleave(low_res_selection, 4, dim=1), 4, dim=2)
    high_res_selection = high_res_selection.reshape(-1, 784)
    x_select = x_in * high_res_selection
    return x_select, high_res_selection

  def _sample_concrete(self, logits):
    # logits: [BATCH_SIZE, d]
    batch_size, n_features = logits.shape
    uniform = pt.rand(batch_size, self.n_key_features, n_features, device=self.device)
    uniform_safe = uniform.clamp(min=np.finfo(np.float32).tiny)

    if self.training:
      gumbel = - pt.log(-pt.log(uniform_safe))
      noisy_logits = (gumbel + logits[:, None, :]) / self.tau
      samples = pt.softmax(noisy_logits, dim=-1)
      samples, _ = pt.max(samples, dim=1)
      return samples
    else:
      # Explanation Stage output.
      threshold = pt.topk(logits, self.n_key_features, sorted=True)[0][:, -1]
      discrete_logits = (logits >= threshold[:, None]).to(pt.float32)
      return discrete_logits

  def classify_selection(self, x_select):
    x = self.act(self.fc4(x_select))
    x = self.bn4(x)
    x = self.act(self.fc5(x))
    x = self.bn5(x)
    x = self.act(self.fc6(x))
    return x



def train_model(model, learning_rate, n_epochs, train_loader, test_loader, device):
  adam = pt.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)

  filepath = f"models/mnist/model.pt"

  for ep in range(n_epochs):

    model.train(True)
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)

      adam.zero_grad()

      loss = nnf.binary_cross_entropy(model(x_batch), y_batch)

      loss.backward()
      adam.step()

    model.train(False)
    summed_loss = 0
    correct_preds = 0
    n_tested = 0
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1)

      preds = model(x_batch)
      loss = nnf.binary_cross_entropy(preds, y_batch)

      summed_loss += loss.item() * y_batch.shape[0]
      class_pred = pt.round(preds)
      correct_preds += pt.sum(class_pred == y_batch).item()
      n_tested += y_batch.shape[0]

    print(f'epoch {ep} done. Acc: {correct_preds / n_tested}, Loss: {summed_loss / n_tested}')

  pt.save(model.state_dict(), filepath)


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
  parser.add_argument('--lr', type=float, default=3e-5)
  parser.add_argument('--no-cuda', action='store_true', default=False)

  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  parser.add_argument('--select-k', type=int, default=4)

  parser.add_argument('--seed', type=int, default=5)
  return parser.parse_args()


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  # train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  train_loader, test_loader = load_two_label_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size,
                                                        data_path='../../data',
                                                        label_a=ar.label_a, label_b=ar.label_b)

  model = L2XModel(d_in=784, d_out=1,
                   datatype='XOR', n_key_features=ar.select_k,
                   d_hid_sel=250, d_hid_clf=500,
                   device=device).to(device)
  train_model(model, ar.lr, ar.epochs, train_loader, test_loader, device)
  # , ar.point_estimate, ar.n_switch_samples, ar.alpha_0, n_features, n_data, ar.KL_reg)

  print('Finished Training Selector')
  x_ts, y_ts, ts_selection = l2x_select_data(model, test_loader, device)
  x_ts_select = x_ts * ts_selection
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
  main()

  # K=5, S=1: 92.0
  # K=5, S=2: 79.1
  # K=5, S=3: 89.4
  # K=5, S=4: 87.1
  # K=5, S=5: 84.6

  # K=4, S=1: 93.5
  # K=4, S=2: 78.8
  # K=4, S=3: 92.0
  # K=4, S=4: 82.6
  # K=4, S=5: 91.6

  # K=3, S=1: 85.3
  # K=3, S=2: 76.2
  # K=3, S=3: 91.5
  # K=3, S=4: 75.3
  # K=3, S=5: 91.5

  # K=2, S=1: 84.7
  # K=2, S=2: 76.3
  # K=2, S=3: 76.3
  # K=2, S=4: 76.3
  # K=2, S=5: 67.0

  # K=1, S=1: 75.2
  # K=1, S=2: 75.2
  # K=1, S=3: 55.3
  # K=1, S=4: 55.3
  # K=1, S=5: 55.3

  # k5_res = [92.0, 79.1, 89.4, 87.1, 84.6]
  # k4_res = [93.5, 78.8, 92.0, 82.6, 91.6]
  # k3_res = [85.3, 76.2, 91.5, 75.3, 91.5]
  # k2_res = [84.7, 76.3, 76.3, 76.3, 67.0]
  # k1_res = [75.2, 75.2, 55.3, 55.3, 55.3]
  #
  # print('k1_avg =', sum(k1_res) / 5)
  # print('k2_avg =', sum(k2_res) / 5)
  # print('k3_avg =', sum(k3_res) / 5)
  # print('k4_avg =', sum(k4_res) / 5)
  # print('k5_avg =', sum(k5_res) / 5)