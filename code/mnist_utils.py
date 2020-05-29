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
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
from switch_model_wrapper import SwitchNetWrapper, BinarizedMnistNet, MnistPatchSelector
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.optim.lr_scheduler import StepLR



class BinarizedMnistDataset(Dataset):
  def __init__(self, train, label_a=3, label_b=8, data_path='../data', download=False):
    super(BinarizedMnistDataset, self).__init__()
    self.train = train
    base_data = datasets.MNIST(data_path, train=train, download=download)

    ids_a, ids_b = base_data.targets == label_a, base_data.targets == label_b
    smp_a, smp_b = base_data.data[ids_a], base_data.data[ids_b]
    n_a, n_b = smp_a.shape[0], smp_b.shape[0]
    print(n_a, n_b)
    tgt_a, tgt_b = base_data.targets[ids_a], base_data.targets[ids_b]
    pert = np.random.permutation(n_a + n_b)
    tgt = np.concatenate([np.zeros(tgt_a.shape), np.ones(tgt_b.shape)])[pert]
    smp = np.concatenate([smp_a, smp_b])[pert]

    smp = np.reshape(smp, (-1, 784))
    self.tgt = tgt.astype(np.float32)
    self.smp = smp.astype(np.float32) / 255

  def __len__(self):
    return self.tgt.shape[0]

  def __getitem__(self, idx):
    return pt.tensor(self.smp[idx]), pt.tensor(self.tgt[idx])

def load_mnist_data(use_cuda, batch_size, test_batch_size, data_path='../data', ):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  train_loader = pt.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([transform]))
  test_loader = pt.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def load_two_label_mnist_data(use_cuda, batch_size, test_batch_size, data_path='../data', label_a=3, label_b=8):
  # select only samples of labels a and b, then binarize labels as a=0 and b=1

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_data = BinarizedMnistDataset(train=True, label_a=label_a, label_b=label_b, data_path=data_path)
  test_data = BinarizedMnistDataset(train=False, label_a=label_a, label_b=label_b, data_path=data_path)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def switch_select_data(selector, loader, device):
    x_data, y_data, selection = [], [], []
    with pt.no_grad():
      for x_tr, y_tr in loader:
        x_data.append(x_tr.numpy())
        y_data.append(y_tr.numpy())
        x_sel = nnf.softplus(selector(x_tr.to(device)))
        x_sel = x_sel / pt.sum(x_sel, dim=1)[:, None] * 16  # multiply by patch size
        selection.append(x_sel.cpu().numpy())
    return np.concatenate(x_data), np.concatenate(y_data), np.concatenate(selection)


def hard_select_data(data, selection, k=1, baseline_val=0):
  effective_k = 16 * k
  sorted_selection = np.sort(selection, axis=1)
  threshold_by_sample = sorted_selection[:, -effective_k]

  below_threshold = selection < threshold_by_sample[:, None]
  feats_selected = 784 - np.sum(below_threshold, axis=1)
  assert np.max(feats_selected) == float(effective_k)
  assert np.max(feats_selected) == np.min(feats_selected)
  data_to_select = np.copy(data)
  data_to_select[below_threshold] = baseline_val

  # make sure no sample has more nonzero entries than allowed in the selection
  assert np.max(np.sum(data_to_select != baseline_val, axis=1)) <= float(effective_k)
  return data_to_select


def make_select_loader(x_select, y_select, train, batch_size, use_cuda, data_path='../data'):
  train_data = BinarizedMnistDataset(train=train, data_path=data_path)
  train_data.smp, train_data.tgt = x_select, y_select
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  return data_loader


def plot_switches(switch_mat, n_rows, n_cols, save_path):
  # normalize to fit the plot
  switch_mat = switch_mat - np.min(switch_mat, axis=1)[:, None]
  switch_mat = switch_mat / np.max(switch_mat, axis=1)[:, None]
  print(np.min(switch_mat), np.max(switch_mat))

  bs = switch_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(switch_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows * i:n_rows * (i + 1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  plt.imsave(save_path + '.png', mnist_mat_flat, cmap=cm.gray, vmin=0., vmax=1.)


def train_classifier(classifier, train_loader, test_loader, epochs, lr, device, lr_decay=None):
  optimizer = optim.Adam(classifier.parameters(), lr=lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay) if lr_decay is not None else None

  for epoch in range(epochs):  # loop over the dataset multiple times
    classifier.train()
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()

      loss = nnf.binary_cross_entropy_with_logits(classifier(x_batch), y_batch)
      # loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg)
      loss.backward()
      optimizer.step()

    if scheduler is not None:
      scheduler.step()
    accuracy = test_classifier_epoch(classifier, test_loader, device, epoch)
  return accuracy


def test_classifier_epoch(classifier, test_loader, device, epoch=-1):
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

  accuracy = correct / len(test_loader.dataset)
  print('Epoch {} Test Loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
    epoch, test_loss, correct, len(test_loader.dataset), 100. * accuracy))
  return accuracy
