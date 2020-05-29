"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

import argparse
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
# from switch_model_wrapper import SwitchNetWrapper, BinarizedMnistNet, MnistPatchSelector
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from switch_mnist_featimp import BinarizedMnistDataset, load_two_label_mnist_data, switch_select_data, \
  hard_select_data, make_select_loader


class L2XModel(nn.Module):
  def __init__(self, d_in, d_out, datatype, n_key_features, device, tau=0.1):
    super(L2XModel, self).__init__()

    self.act = nn.ReLU() if datatype in ['orange_skin', 'XOR'] else nn.SELU()
    self.d_out = d_out

    # q(S|X)
    self.fc1 = nn.Linear(d_in, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 49)

    # concrete sampling
    self.tau = tau
    self.n_key_features = n_key_features
    self.device = device

    # q(X_S)
    self.fc4 = nn.Linear(d_in, 200)
    self.bn4 = nn.BatchNorm1d(200)
    self.fc5 = nn.Linear(200, 200)
    self.bn5 = nn.BatchNorm1d(200)
    self.fc6 = nn.Linear(200, d_out)

  def forward(self, x_in):
    x_select, _ = self.get_selection(x_in)
    pred = self.classify_selection(x_select)

    if self.d_out == 1:
      pred = pt.sigmoid(pred.view(-1))
    return pred

  def get_selection(self, x_in):
    x = self.act(self.fc1(x_in))
    x = self.act(self.fc2(x))
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



def train_model(model, selected_label, learning_rate, n_epochs, train_loader, test_loader, device):
  adam = pt.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)

  filepath = f"models/mnist/model.pt"

  for ep in range(n_epochs):

    model.train(True)
    for x_batch, y_batch in train_loader:
      x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)
      y_batch = (y_batch == selected_label).to(pt.float32).to(device)
      adam.zero_grad()

      loss = nnf.binary_cross_entropy(model(x_batch), y_batch)

      loss.backward()
      adam.step()

    model.train(False)
    summed_loss = 0
    correct_preds = 0
    n_tested = 0
    for x_batch, y_batch in test_loader:
      x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)
      y_batch = (y_batch == selected_label).to(pt.float32).to(device)


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

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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


def train_selector(model, train_loader, epochs, lr, device):
  # , point_estimate, n_switch_samples, alpha_0, n_features, n_data, KL_reg):
  optimizer = optim.Adam(model.selector_params(), lr=lr)
  training_loss_per_epoch = np.zeros(epochs)
  # annealing_steps = float(8000. * epochs)

  # def beta_func(s):
  #   return min(s, annealing_steps) / annealing_steps

  for epoch in range(epochs):  # loop over the dataset multiple times
    print(f'epoch {epoch}')
    running_loss = 0.0

    # annealing_rate = beta_func(epoch)

    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs, phi_cand = model(x_batch)  # 100,10,150

      loss = nnf.binary_cross_entropy_with_logits(outputs, y_batch)
      # loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()

    # training_loss_per_epoch[epoch] = running_loss/how_many_samps
    training_loss_per_epoch[epoch] = running_loss
    print('epoch number is ', epoch)
    print('running loss is ', running_loss)


def local_switch_eval(model, test_loader):
  pass


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=2)
  parser.add_argument('--dataset', type=str, default='mnist')
  # parser.add_argument('--selected-label', type=int, default=3)  # label for 1-v-rest training
  # parser.add_argument('--log-interval', type=int, default=500)
  parser.add_argument('--n-switch-samples', type=int, default=10)

  parser.add_argument('--save-model', action='store_true', default=False)
  parser.add_argument("--point_estimate", default=True)
  # parser.add_argument("--KL_reg", default=False)
  # parser.add_argument('--alpha_0', type=float, default=50000.)
  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  parser.add_argument('--select-k', type=int, default=4)

  # parser.add_argument("--freeze-classifier", default=True)
  # parser.add_argument("--patch-selection", default=True)

  return parser.parse_args()


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  # train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  train_loader, test_loader = load_two_label_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size,
                                                        label_a=ar.label_a, label_b=ar.label_b)
  # unpack data
  n_data, n_features = 60000, 784

  classifier = BinarizedMnistNet().to(device)
  # classifier.load_state_dict(pt.load(f'models/{ar.dataset}_nn_ep4.pt'))
  train_classifier(classifier, train_loader, test_loader, ar.epochs, ar.lr, device)
  print('Finished Training Classifier')

  # selector = MnistPatchSelector().to(device)
  # model = SwitchNetWrapper(selector, classifier, n_features, ar.n_switch_samples, ar.point_estimate).to(device)

  model = L2XModel(d_in=784, d_out=1, datatype=None, n_key_features=n_select, device=device).to(device)


  train_model(model, selected_label, learning_rate, n_epochs, train_loader, test_loader, device)
  st2 = time.time()

  train_selector(model, train_loader, ar.epochs, ar.lr, device)
  # , ar.point_estimate, ar.n_switch_samples, ar.alpha_0, n_features, n_data, ar.KL_reg)
  print('Finished Training Selector')



  x_tr, y_tr, tr_selection = switch_select_data(selector, train_loader, device)
  x_ts, y_ts, ts_selection = switch_select_data(selector, test_loader, device)
  x_tr_select = hard_select_data(x_tr, tr_selection, k=ar.select_k)
  x_ts_select = hard_select_data(x_ts, ts_selection, k=ar.select_k)
  select_train_loader = make_select_loader(x_tr_select, y_tr, train=True, batch_size=ar.batch_size, use_cuda=use_cuda)
  select_test_loader = make_select_loader(x_ts_select, y_ts, train=False, batch_size=ar.test_batch_size,
                                          use_cuda=use_cuda)
  print('testing classifier')
  test_classifier_epoch(classifier, select_test_loader, device)

  # print('testing retrained classifier')
  # new_classifier = BinarizedMnistNet().to(device)
  # train_classifier(new_classifier, select_train_loader, select_test_loader, ar.epochs, ar.lr, device)


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  do_featimp_exp(ar)

if __name__ == '__main__':
  main()
