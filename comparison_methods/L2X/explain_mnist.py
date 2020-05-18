import numpy as np
# import tensorflow as tf
# import pandas as pd
# import cPickle as pkl
# from collections import defaultdict
# import re
# from bs4 import BeautifulSoup
# import sys
# import os
import time
# import json
import random
import argparse

import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# from switch_mnist_featimp import load_mnist_data
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from L2X.explain_pytorch import L2XModel
except ImportError:
  from explain_pytorch import L2XModel

def load_mnist_data(use_cuda, batch_size, test_batch_size, data_path='../data'):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  train_loader = pt.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([transform]))
  test_loader = pt.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


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


def test_model(model, test_loader, device, selected_label, vis_only_selected, save_path):
  model.train(False)

  scores = []

  for x_batch, y_batch in test_loader:
    x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)
    if vis_only_selected:
      x_batch = x_batch[y_batch.to(device) == selected_label]

    scores.append(model.get_selection(x_batch)[1].cpu().numpy())

  scores = np.concatenate(scores, axis=0)
  # median_ranks = compute_median_rank(scores, k=n_key_features, datatype_val=datatype_val)
  scores_avg = np.sum(scores, axis=0)
  scores_avg = scores_avg / np.max(scores_avg)

  mnist_mat = np.reshape(scores_avg, (28, 28))

  plt.imsave(save_path + '.png', mnist_mat, cmap=cm.gray, vmin=0., vmax=1.)


def L2X(batch_size, selected_label, n_select, n_epochs, learning_rate, use_cuda, skip_training, vis_only_selected):
  # x_train, y_train, x_test, y_test, datatype_val, input_shape = create_data(datatype, n=int(1e6))
  device = pt.device("cuda" if use_cuda else "cpu")
  train_loader, test_loader = load_mnist_data(use_cuda, batch_size, batch_size, data_path='../../data')
  st1 = time.time()
  st2 = st1

  model = L2XModel(d_in=784, d_out=1, datatype=None, n_key_features=n_select, device=device).to(device)

  if not skip_training:
    train_model(model, selected_label, learning_rate, n_epochs, train_loader, test_loader, device)
    st2 = time.time()
  else:
    model.load_state_dict(pt.load(f'models/mnist/model.pt'))

  vis_save_path = f'models/mnist/vis_{"only_selected_" if vis_only_selected else ""}label{selected_label}_top{n_select}'
  test_model(model, test_loader, device, selected_label, vis_only_selected, vis_save_path)

  print(f'train time:{st2 - st1}s, explain time:{time.time() - st2}s')


def main():
  # The number of key features for each data set.

  parser = argparse.ArgumentParser()
  parser.add_argument('--skip_training', action='store_true', default=False)
  parser.add_argument('--batch-size', type=int, default=100)
  parser.add_argument('--lr', type=int, default=1e-3)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--n-select', type=int, default=5)
  parser.add_argument('--selected-label', type=int, default=0)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--vis-only-selected', action='store_true', default=False)
  ar = parser.parse_args()

  use_cuda = not ar.no_cuda and pt.cuda.is_available()

  np.random.seed(ar.seed)
  random.seed(ar.seed)
  pt.manual_seed(ar.seed)

  L2X(ar.batch_size, ar.selected_label, ar.n_select, ar.epochs, ar.lr, use_cuda, ar.skip_training, ar.vis_only_selected)

  # print(f'data type: {ar.datatype}')
  # print(f'mean:{np.mean(median_ranks)}, sd:{np.std(median_ranks)}')
  # print(f'train time:{train_time}s, explain time:{exp_time}s')


if __name__ == '__main__':
  main()
