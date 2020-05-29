"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'frederik'

import argparse
import numpy as np

import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Gamma
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from switch_model_wrapper import SwitchNetWrapper, BinarizedMnistNet, MnistPatchSelector
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mnist_utils import plot_switches, load_two_label_mnist_data, switch_select_data, hard_select_data, \
  test_classifier_epoch, make_select_loader, train_classifier, BinarizedMnistDataset
import os


class BinarizedTestClassifier(nn.Module):
  def __init__(self, d_hid=300, label_a=None, label_b=None, model_path_prefix=''):
    super(BinarizedTestClassifier, self).__init__()
    default_dir = 'models/bin_mnist_classifiers/'
    self.model_dir = os.path.join(model_path_prefix, default_dir)

    self.fc1 = nn.Linear(784, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.fc3 = nn.Linear(d_hid, 1)
    self.bn1 = nn.BatchNorm1d(d_hid)
    self.bn2 = nn.BatchNorm1d(d_hid)

    self.label_a = label_a
    self.label_b = label_b
    if label_a is not None and label_b is not None:
      self.load_stored_weights()

  def forward(self, x):
    x = pt.flatten(x, 1)
    x = self.fc1(x)
    x = self.bn1(x)
    x = nnf.relu(x)
    x = self.fc2(x)
    x = self.bn2(x)
    x = nnf.relu(x)
    x = self.fc3(x)
    x = x.flatten()
    return x  # assume BCE with logits as loss

  def load_stored_weights(self):
    assert self.label_a < self.label_b
    assert self.label_a != 0 or self.label_b != 6
    load_file = os.path.join(self.model_dir, f'bin_classifier_a{self.label_a}_b{self.label_b}.pt')
    self.load_state_dict(pt.load(load_file))


def train_test_classifier(label_a, label_b, epochs, lr, lr_decay, data_path='../data',
                          batch_size=64, test_batch_size=1000, seed=1):
  assert label_a < label_b  # prevent ordering mixups
  np.random.seed(seed)
  pt.manual_seed(seed)

  train_data = BinarizedMnistDataset(train=True, label_a=label_a, label_b=label_b, data_path=data_path)
  test_data = BinarizedMnistDataset(train=False, label_a=label_a, label_b=label_b, data_path=data_path)

  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

  classifier = BinarizedTestClassifier().to(device)
  train_classifier(classifier, train_loader, test_loader, epochs, lr, device, lr_decay)

  print('training done - saving model')
  os.makedirs(classifier.model_dir, exist_ok=True)
  pt.save(classifier.state_dict(), f'bin_classifier_a{label_a}_b{label_b}.pt')


def test_posthoc_acc(label_a, label_b, test_loader, device, model_path_prefix):
  classifier = BinarizedTestClassifier(label_a=label_a, label_b=label_b, model_path_prefix=model_path_prefix).to(device)
  accuracy = test_classifier_epoch(classifier, test_loader, device, epoch=-1)
  return accuracy


def train_models():
  for l_b in range(1, 6):
    for l_a in range(l_b):
      if l_a == 0 and l_b == 6:
        print(f'skipping 0,6 because batchnorm breaks due to train set size')
        continue
      print(f'running a{l_a}, b{l_b}')
      train_test_classifier(label_a=l_a, label_b=l_b, epochs=30, lr=3e-4, lr_decay=0.9)


if __name__ == '__main__':
  train_models()
