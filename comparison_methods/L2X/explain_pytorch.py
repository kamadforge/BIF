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
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from L2X.make_data import generate_data


def create_data(datatype, n=1000):
  """
  Create train and validation datasets.

  """
  x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
  x_val, y_val, datatypes_val = generate_data(n=10 ** 5, datatype=datatype, seed=1)

  input_shape = x_train.shape[1]

  return x_train, y_train, x_val, y_val, datatypes_val, input_shape


def create_rank(scores, k):
  """
  Compute rank of each feature based on weight.

  """
  scores = abs(scores)
  n, d = scores.shape
  ranks = []
  for i, score in enumerate(scores):
    # Random permutation to avoid bias due to equal weights.
    idx = np.random.permutation(d)
    permutated_weights = score[idx]
    permutated_rank = (-permutated_weights).argsort().argsort() + 1
    rank = permutated_rank[np.argsort(idx)]

    ranks.append(rank)

  return np.array(ranks)


def compute_median_rank(scores, k, datatype_val=None):
  ranks = create_rank(scores, k)
  if datatype_val is None:
    median_ranks = np.median(ranks[:, :k], axis=1)
  else:
    datatype_val = datatype_val[:len(scores)]
    median_ranks1 = np.median(ranks[datatype_val == 'orange_skin', :][:, np.array([0, 1, 2, 3, 9])], axis=1)
    median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive', :][:, np.array([4, 5, 6, 7, 9])], axis=1)
    median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
  return median_ranks


# class Sample_Concrete(Layer):
#   """
#   Layer for sample Concrete / Gumbel-Softmax variables.
#
#   """
#
#   def __init__(self, tau0, k, **kwargs):
#     self.tau0 = tau0
#     self.k = k
#     super(Sample_Concrete, self).__init__(**kwargs)
#
#   def call(self, logits):
#     # logits: [BATCH_SIZE, d]
#     logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]
#
#     batch_size = tf.shape(logits_)[0]
#     d = tf.shape(logits_)[2]
#     uniform = tf.random_uniform(shape=(batch_size, self.k, d), minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
#                                 maxval=1.0)
#
#     gumbel = - K.log(-K.log(uniform))
#     noisy_logits = (gumbel + logits_) / self.tau0
#     samples = K.softmax(noisy_logits)
#     samples = K.max(samples, axis=1)
#
#     # Explanation Stage output.
#     threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
#     discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
#
#     return K.in_train_phase(samples, discrete_logits)
#
#   def compute_output_shape(self, input_shape):
#     return input_shape


class L2XModel(nn.Module):
  def __init__(self, d_in, d_out, datatype, n_key_features, device, tau=0.1):
    super(L2XModel, self).__init__()

    self.act = nn.ReLU() if datatype in ['orange_skin', 'XOR'] else nn.SELU()

    # q(S|X)
    self.fc1 = nn.Linear(d_in, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, d_in)

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
    return pred

  def get_selection(self, x_in):
    x = self.act(self.fc1(x_in))
    x = self.act(self.fc2(x))
    x = self.fc3(x)

    selection = self._sample_concrete(x)

    x_select = x_in * selection
    return x_select, selection

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


def train_model(model, datatype, batch_size, learning_rate, n_epochs, x_train, y_train, x_test, y_test, device):
  adam = pt.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)
  loss_fun = nn.CrossEntropyLoss()
  filepath = f"models/{datatype}/model.pt"

  n_train_steps = int(np.ceil(x_train.shape[0] / batch_size))
  n_test_steps = int(np.ceil(x_test.shape[0] / batch_size))

  for ep in range(n_epochs):
    # shuffle dataset
    train_perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[train_perm]
    y_train = y_train[train_perm]

    # training
    model.train(True)
    for step in range(n_train_steps):
      adam.zero_grad()

      x_batch = x_train[step * batch_size: (step + 1) * batch_size]
      y_batch = y_train[step * batch_size: (step + 1) * batch_size]
      x_batch = pt.tensor(x_batch, device=device)
      y_batch = pt.tensor(y_batch, device=device)

      loss = loss_fun(model(x_batch), y_batch)

      loss.backward()
      adam.step()

    model.train(False)
    summed_loss = 0
    correct_preds = 0
    for step in range(n_test_steps):
      x_batch = x_test[step * batch_size: (step + 1) * batch_size]
      y_batch = y_test[step * batch_size: (step + 1) * batch_size]

      x_batch = pt.tensor(x_batch, device=device)
      y_batch = pt.tensor(y_batch, device=device)

      preds = model(x_batch)
      loss = loss_fun(preds, y_batch)

      summed_loss += loss.item() * y_batch.shape[0]
      class_pred = pt.max(preds, dim=1)[1]
      correct_preds += pt.sum(class_pred == y_batch).item()

    n_test = y_test.shape[0]
    print(f'epoch {ep} done. Acc: {correct_preds / n_test}, Loss: {summed_loss / n_test}')

  pt.save(model.state_dict(), filepath)


def test_model(model, x_test, datatype, batch_size, device, n_key_features, datatype_val):
  model.train(False)
  n_test_steps = int(np.ceil(x_test.shape[0] / batch_size))
  scores = []

  for step in range(n_test_steps):
    x_batch = x_test[step * batch_size: (step + 1) * batch_size]
    # y_batch = y_test[step * batch_size: (step + 1) * batch_size]

    x_batch = pt.tensor(x_batch, device=device)
    # y_batch = pt.tensor(y_batch, device=device)

    scores.append(model.get_selection(x_batch)[1].cpu().numpy())

  scores = np. concatenate(scores, axis=0)
  median_ranks = compute_median_rank(scores, k=n_key_features, datatype_val=datatype_val)


  print(f'data type: {datatype}')
  print(f'mean:{np.mean(median_ranks)}, sd:{np.std(median_ranks)}')


def discretize_labels(y):
  one_hot = (y == np.max(y, axis=1)[:, None]).astype(np.int)
  return np.argmax(one_hot, axis=1)



def L2X(datatype, batch_size, n_key_features, n_epochs, learning_rate, device, skip_training):
  x_train, y_train, x_test, y_test, datatype_val, input_shape = create_data(datatype, n=int(1e6))

  x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
  y_train, y_test = discretize_labels(y_train), discretize_labels(y_test)
  print(np.sum(y_test))
  print(x_train.shape, y_train.shape)

  st1 = time.time()
  st2 = st1

  model = L2XModel(d_in=x_train.shape[1], d_out=2, datatype=datatype, n_key_features=n_key_features,
                   device=device).to(device)

  if not skip_training:
    train_model(model, datatype, batch_size, learning_rate, n_epochs, x_train, y_train, x_test, y_test, device)
    st2 = time.time()
  else:
    model.load_state_dict(pt.load(f'models/{datatype}/model.pt'))

  test_model(model, x_test, datatype, batch_size, device, n_key_features, datatype_val)

  print(f'train time:{st2 - st1}s, explain time:{time.time() - st2}s')


def main():
  # The number of key features for each data set.
  n_key_features_by_data = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5}

  parser = argparse.ArgumentParser()
  parser.add_argument('--datatype', type=str, choices=['orange_skin', 'XOR', 'nonlinear_additive', 'switch'],
                      default='nonlinear_additive')
  parser.add_argument('--skip_training', action='store_true', default=False)
  parser.add_argument('--batch-size', type=int, default=1000)
  parser.add_argument('--lr', type=int, default=1e-3)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  ar = parser.parse_args()

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  np.random.seed(ar.seed)
  random.seed(ar.seed)
  pt.manual_seed(ar.seed)

  n_key_features = n_key_features_by_data[ar.datatype]
  L2X(ar.datatype, ar.batch_size, n_key_features, ar.epochs, ar.lr, device, ar.skip_training)

  # print(f'data type: {ar.datatype}')
  # print(f'mean:{np.mean(median_ranks)}, sd:{np.std(median_ranks)}')
  # print(f'train time:{train_time}s, explain time:{exp_time}s')


if __name__ == '__main__':
  main()
