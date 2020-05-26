from __future__ import print_function
import numpy as np
import tensorflow as tf
# import pandas as pd
# import cPickle as pkl
# from collections import defaultdict
# import re
# from bs4 import BeautifulSoup
# import sys
import os
import time
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Multiply  # , Flatten, Add, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model  # , Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
# from make_data import generate_data
# import json
import random
from keras import optimizers


try:
  from invase_imports import generate_dataset, feature_performance_metric, prediction_performance_metric
except ImportError:
  from invase_imports import generate_dataset, feature_performance_metric, prediction_performance_metric

BATCH_SIZE = 1000
# np.random.seed(0)
# tf.set_random_seed(0)
# random.seed(0)
# The number of key features for each data set.
ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5,
      'syn1': 2, 'syn2': 4, 'syn3': 4}


# def create_data(datatype, n=1000):
#   """
#   Create train and validation datasets.
#
#   """
#   x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
#   x_val, y_val, datatypes_val = generate_data(n=10 ** 5, datatype=datatype, seed=1)
#
#   input_shape = x_train.shape[1]
#
#   return x_train, y_train, x_val, y_val, datatypes_val, input_shape

def create_data(datatype, n=1000):
  """
  Create train and validation datasets.

  """
  x_train, y_train, _ = generate_dataset(n=n, data_type=datatype, seed=0)
  x_val, y_val, ground_truth_val = generate_dataset(n=n//10, data_type=datatype, seed=1)

  input_shape = x_train.shape[1]

  return x_train, y_train, x_val, y_val, ground_truth_val, input_shape


def load_data(dataset_name):
  ###########################################
  # LOAD DATA

  x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset_name)
  y_tot = np.stack([1 - y_tot, y_tot], axis=1)

  # unpack data
  N_tot, n_feats = x_tot.shape

  training_data_por = 0.8
  N = int(training_data_por * N_tot)


  if dataset_name == "alternating" or "syn" in dataset_name:
    datatypes = datatypes_tot[:N]  # only for alternating, if datatype comes from orange_skin or nonlinear
    datatypes_test = datatypes_tot[N:]
  elif dataset_name == "XOR":
    datatypes = None
    datatypes_test = np.zeros((N_tot - N, n_feats))
    datatypes_test[:, :2] = 1.
  elif dataset_name in {"orange_skin", "nonlinear_additive"}:
    datatypes = None
    datatypes_test = np.zeros((N_tot - N, n_feats))
    datatypes_test[:, :4] = 1.
  else:
    raise ValueError

  y_train, x_train, datatypes_train = shuffle_data(y_tot[:N], x_tot[:N, :], N, datatypes)
  x_test = x_tot[N:, :]
  y_test = y_tot[N:]


  return x_train, y_train, x_test, y_test, datatypes_test, n_feats


def shuffle_data(y, x, how_many_samps, datatypes=None):

    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]
    if datatypes is None:
        shuffled_datatypes = None
    else:
        shuffled_datatypes = datatypes[idx]

    return shuffled_y, shuffled_x, shuffled_datatypes


def synthetic_data_loader(dataset_name):

  sub_dir = 'invase' if dataset_name.startswith('syn') else dataset_name
  data_file = '../../data/synthetic/' + sub_dir + '/dataset_' + dataset_name + '.npy'
  loaded_dataset = np.load(data_file, allow_pickle=True)

  x_tot = loaded_dataset[()]['x']
  y_tot = loaded_dataset[()]['y']

  if dataset_name in {'alternating', 'syn4', 'syn5', 'syn6'}:
    datatypes = loaded_dataset[()]['datatypes']
  else:
    datatypes = None

  return x_tot, y_tot, datatypes

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


def invase_style_analysis(scores, gt_test, preds, y_test):

  importance_score = 1. * (scores > 0.5)

  # Evaluate the performance of feature importance
  mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(gt_test, importance_score)

  # Print the performance of feature importance
  print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '%, ' + 'TPR std: ' + str(np.round(std_tpr, 1)) + '%, ')
  print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '%, ' + 'FDR std: ' + str(np.round(std_fdr, 1)) + '%, ')

  # Predict labels
  # Evaluate the performance of feature importance
  auc, apr, acc = prediction_performance_metric(y_test, preds)

  # Print the performance of feature importance
  print('AUC: ' + str(np.round(auc, 3)) + ', APR: ' + str(np.round(apr, 3)) + ', ACC: ' + str(np.round(acc, 3)))
  performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr, 'mean_fdr': mean_fdr,
                 'std_fdr': std_fdr, 'auc': auc, 'apr': apr, 'acc': acc}
  return performance


class Sample_Concrete(Layer):
  """
  Layer for sample Concrete / Gumbel-Softmax variables.

  """

  def __init__(self, tau0, k, **kwargs):
    self.tau0 = tau0
    self.k = k
    super(Sample_Concrete, self).__init__(**kwargs)

  def call(self, logits):
    # logits: [BATCH_SIZE, d]
    logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

    batch_size = tf.shape(logits_)[0]
    d = tf.shape(logits_)[2]
    uniform = tf.random_uniform(shape=(batch_size, self.k, d), minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                maxval=1.0)

    gumbel = - K.log(-K.log(uniform))
    noisy_logits = (gumbel + logits_) / self.tau0
    samples = K.softmax(noisy_logits)
    samples = K.max(samples, axis=1)

    # Explanation Stage output.
    threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
    discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

    return K.in_train_phase(samples, discrete_logits)

  def compute_output_shape(self, input_shape):
    return input_shape


def L2X(datatype, train=True):
  # x_train, y_train, x_val, y_val, ground_truth_val, input_shape = create_data(datatype, n=int(n_data))
  x_train, y_train, x_val, y_val, ground_truth_val, input_shape = load_data(datatype)

  st1 = time.time()
  st2 = st1

  # activation = 'relu' if datatype in ['syn1', 'syn2'] else 'selu'
  activation = 'relu' if datatype in {'syn1', 'syn2', 'syn3', 'XOR', 'orange_skin'} else 'selu'
  # P(S|X)
  model_input = Input(shape=(input_shape,), dtype='float32')

  net = Dense(100, activation=activation, name='s/dense1', kernel_regularizer=regularizers.l2(1e-3))(model_input)
  net = Dense(100, activation=activation, name='s/dense2', kernel_regularizer=regularizers.l2(1e-3))(net)

  # A tensor of shape, [batch_size, max_sents, 100]
  logits = Dense(input_shape)(net)
  # [BATCH_SIZE, max_sents, 1]
  k = ks[datatype]
  tau = 0.1
  samples = Sample_Concrete(tau, k, name='sample')(logits)

  # q(X_S)
  new_model_input = Multiply()([model_input, samples])
  net = Dense(200, activation=activation, name='dense1', kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
  net = BatchNormalization()(net)  # Add batchnorm for stability.
  net = Dense(200, activation=activation, name='dense2', kernel_regularizer=regularizers.l2(1e-3))(net)
  net = BatchNormalization()(net)

  preds = Dense(2, activation='softmax', name='dense4', kernel_regularizer=regularizers.l2(1e-3))(net)
  model = Model(model_input, preds)

  if train:
    adam = optimizers.Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    base_dir = "models/{}/".format(datatype)
    if not os.path.exists(base_dir):
      os.makedirs(base_dir)
    filepath = base_dir + "L2X.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # print(y_train.shape)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks_list, epochs=125,
              batch_size=BATCH_SIZE, verbose=0)
    st2 = time.time()
  else:
    model.load_weights('models/{}/L2X.hdf5'.format(datatype), by_name=True)

  selection_model = Model(model_input, samples)
  selection_model.compile(loss=None, optimizer='rmsprop', metrics=None)

  # prediction_model = Model(new_model_input, preds)
  # prediction_model.compile(loss=None, optimizer='rmsprop', metrics=None)

  scores = selection_model.predict(x_val, verbose=0, batch_size=BATCH_SIZE)
  # test_preds = model.predict(x_val * scores, verbose=1, batch_size=BATCH_SIZE)
  test_preds = model.predict(x_val, verbose=0, batch_size=BATCH_SIZE)
  # print('shapes:', scores.shape, ground_truth_val.shape, test_preds.shape, y_val.shape)
  # print('sums:', np.sum(scores), np.sum(ground_truth_val), np.sum(test_preds), np.sum(y_val))
  invase_style_analysis(scores, ground_truth_val, test_preds, y_val)
  # median_ranks = compute_median_rank(scores, k=ks[datatype], datatype_val=None)

  # print('median ranks mean:', np.mean(median_ranks), ' std: ', np.std(median_ranks))

  return time.time() - st2, st2 - st1


def run_experiment(datatype, seed, custom_k, train):
  print('running experiment with params:', datatype, seed, custom_k, train)
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  if datatype in {'syn4', 'syn5', 'syn6'}:
    ks[datatype] = custom_k

  exp_time, train_time = L2X(datatype=datatype, train=train)
  output = 'datatype:{} train time:{}s, explain time:{}s \n'.format(datatype, train_time, exp_time)
  print(output)


def main():
  import argparse

  parser = argparse.ArgumentParser()
  data_options = ['XOR', 'orange_skin', 'nonlinear_additive',
                  'syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6']
  parser.add_argument('--datatype', type=str, choices=data_options, default='syn4')
  parser.add_argument('--train', action='store_true', default=True)
  parser.add_argument('--custom_k', type=int, default=4)
  # parser.add_argument('--n_data', type=int, default=1e6)
  parser.add_argument('--seed', type=int, default=1)
  args = parser.parse_args()

  run_experiment(args.datatype, args.seed, args.custom_k, args.train)


def syn_test_full_run():
  run_experiment('XOR', seed=1, custom_k=None, train=True)
  run_experiment('XOR', seed=2, custom_k=None, train=True)
  run_experiment('XOR', seed=3, custom_k=None, train=True)
  run_experiment('XOR', seed=4, custom_k=None, train=True)
  run_experiment('XOR', seed=5, custom_k=None, train=True)

  run_experiment('orange_skin', seed=1, custom_k=None, train=True)
  run_experiment('orange_skin', seed=2, custom_k=None, train=True)
  run_experiment('orange_skin', seed=3, custom_k=None, train=True)
  run_experiment('orange_skin', seed=4, custom_k=None, train=True)
  run_experiment('orange_skin', seed=5, custom_k=None, train=True)

  run_experiment('nonlinear_additive', seed=1, custom_k=None, train=True)
  run_experiment('nonlinear_additive', seed=2, custom_k=None, train=True)
  run_experiment('nonlinear_additive', seed=3, custom_k=None, train=True)
  run_experiment('nonlinear_additive', seed=4, custom_k=None, train=True)
  run_experiment('nonlinear_additive', seed=5, custom_k=None, train=True)

  run_experiment('syn4', seed=1, custom_k=2, train=True)
  run_experiment('syn4', seed=2, custom_k=2, train=True)
  run_experiment('syn4', seed=3, custom_k=2, train=True)
  run_experiment('syn4', seed=4, custom_k=2, train=True)
  run_experiment('syn4', seed=5, custom_k=2, train=True)

  run_experiment('syn4', seed=1, custom_k=3, train=True)
  run_experiment('syn4', seed=2, custom_k=3, train=True)
  run_experiment('syn4', seed=3, custom_k=3, train=True)
  run_experiment('syn4', seed=4, custom_k=3, train=True)
  run_experiment('syn4', seed=5, custom_k=3, train=True)

  run_experiment('syn4', seed=1, custom_k=4, train=True)
  run_experiment('syn4', seed=2, custom_k=4, train=True)
  run_experiment('syn4', seed=3, custom_k=4, train=True)
  run_experiment('syn4', seed=4, custom_k=4, train=True)
  run_experiment('syn4', seed=5, custom_k=4, train=True)

  run_experiment('syn5', seed=1, custom_k=2, train=True)
  run_experiment('syn5', seed=2, custom_k=2, train=True)
  run_experiment('syn5', seed=3, custom_k=2, train=True)
  run_experiment('syn5', seed=4, custom_k=2, train=True)
  run_experiment('syn5', seed=5, custom_k=2, train=True)

  run_experiment('syn5', seed=1, custom_k=3, train=True)
  run_experiment('syn5', seed=2, custom_k=3, train=True)
  run_experiment('syn5', seed=3, custom_k=3, train=True)
  run_experiment('syn5', seed=4, custom_k=3, train=True)
  run_experiment('syn5', seed=5, custom_k=3, train=True)

  run_experiment('syn5', seed=1, custom_k=4, train=True)
  run_experiment('syn5', seed=2, custom_k=4, train=True)
  run_experiment('syn5', seed=3, custom_k=4, train=True)
  run_experiment('syn5', seed=4, custom_k=4, train=True)
  run_experiment('syn5', seed=5, custom_k=4, train=True)

  run_experiment('syn6', seed=1, custom_k=4, train=True)
  run_experiment('syn6', seed=2, custom_k=4, train=True)
  run_experiment('syn6', seed=3, custom_k=4, train=True)
  run_experiment('syn6', seed=4, custom_k=4, train=True)
  run_experiment('syn6', seed=5, custom_k=4, train=True)


if __name__ == '__main__':
  main()
  # syn_test_full_run()