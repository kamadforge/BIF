"""
Test learning global feature importance with/without switch net for Syn1-3 datasets.

"""

__author__ = 'anon_m'
# 28 May 2020

""" import packages """
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import pickle
import argparse
import socket

from pathlib import Path
import sys
import os
import socket
from data.make_synthetic_datasets import generate_data
# from evaluation_metrics import binary_classification_metrics, compute_median_rank
from Models import Feedforward, Feature_Importance_Model
from Losses import loss_function
from sklearn.metrics import roc_auc_score
import shap
import xgboost


max_seed = 1
input_dim = 10
posterior_mean = np.zeros((max_seed,input_dim))

for seed_idx in range(max_seed):

    np.random.seed(seed_idx)

    """ generate data """
    N_tot = 10000
    # dataset = 'XOR'
    # dataset = 'orange_skin'
    dataset = 'nonlinear_additive'
    x_tot, y_tot, datatypes = generate_data(N_tot, dataset)
    y_tot = np.argmax(y_tot, axis=1)

    # train and test data
    N = np.int(N_tot*0.9)
    rand_perm_nums = np.random.permutation(N_tot)
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]
    Xtst = x_tot[rand_perm_nums[N:], :]
    ytst = y_tot[rand_perm_nums[N:]]

    """ train a classifier """
    input_dim = x_tot.shape[1]

    classifier = Feedforward(input_dim, 100, 20)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.05)

    classifier.train()
    how_many_epochs = 10
    # mini_batch_size = 100
    # how_many_iter = np.int(N / mini_batch_size)
    mini_batch_size = N_tot  # generally larger mini batch size is more helpful for switch learning
    how_many_iter = np.int(N / mini_batch_size)
    if mini_batch_size==N_tot:
        how_many_iter = 1

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            inputs = X[i * mini_batch_size:(i + 1) * mini_batch_size, :]
            labels = y[i * mini_batch_size:(i + 1) * mini_batch_size]

            optimizer.zero_grad()
            y_pred = classifier(torch.Tensor(inputs))
            loss = criterion(y_pred.squeeze(), torch.FloatTensor(labels))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        y_pred = classifier(torch.Tensor(Xtst))
        ROC = roc_auc_score(ytst, y_pred.detach().numpy())
        print('Epoch {}: ROC : {}'.format(epoch, ROC))

    print('Finished Classifier Training')


    ###########################################

    """ learn feature importance """


    # we initialize the parameters based on SHAP estimation.
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_sv = np.abs(shap_values).mean(axis=0)
    print('importance by SHAP: ', mean_sv)

    filename = dataset + 'shap.npy'
    np.save(filename, mean_sv)

    # importance by SHAP:  [3.01049769e-01 2.16896809e-03 8.27491842e-03 1.28855770e-02
    #  1.08360124e-04 3.88371904e-04 9.35922362e-05 1.77524707e-04
    #  1.42073390e-04 2.23139068e-04]

    # input_dim, classifier,
    init_phi = 100*mean_sv # Initializing with SHAP estimate, and start with very small concentration

    num_Dir_samps = 10
    importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps, init_phi)
    optimizer = optim.Adam(importance.parameters(), lr=0.075)

    # We freeze the classifier
    ct = 0
    for child in importance.children():
        ct += 1
        if ct >= 1:
            for param in child.parameters():
                param.requires_grad = False

    # print(list(importance.parameters())) # make sure I only update the gradients of feature importance

    importance.train()
    how_many_epochs = 200
    mini_batch_size = N_tot  # generally larger mini batch size is more helpful for switch learning
    how_many_iter = np.int(N / mini_batch_size)

    if mini_batch_size==N_tot:
        how_many_iter = 1


    if dataset=='nonlinear_additive':
        alpha_0 = 0.2
    else:
        alpha_0 = 0.01

    annealing_rate = 1  # we don't anneal. don't want to think about this.
    kl_term = True

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            inputs = X[i * mini_batch_size:(i + 1) * mini_batch_size, :]
            labels = y[i * mini_batch_size:(i + 1) * mini_batch_size]

            optimizer.zero_grad()
            y_pred, phi_cand = importance(torch.Tensor(inputs))
            labels = torch.squeeze(torch.Tensor(labels))
            loss = loss_function(y_pred, labels.view(-1, 1).repeat(1, num_Dir_samps), phi_cand, alpha_0, input_dim,
                                 annealing_rate, N, kl_term)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            estimated_params = list(importance.parameters())
            phi_est = F.softplus(torch.Tensor(estimated_params[0]))
            print('phi_est', phi_est.detach().numpy())

        print('Epoch {}: running_loss : {}'.format(epoch, running_loss))
        #
        # if np.remainder(epoch, 50)==0:
        #     # every 50th epoch, we check the results
        #     """ checking the results """
        #     estimated_params = list(importance.parameters())
        #     phi_est = F.softplus(torch.Tensor(estimated_params[0]))
        #     concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
        #     beta_param = torch.ones(concentration_param.size())
        #     Gamma_obj = Gamma(concentration_param, beta_param)
        #     gamma_samps = Gamma_obj.rsample()
        #     Sstack = gamma_samps / torch.sum(gamma_samps, 0)
        #     avg_S = torch.mean(Sstack, 1)
        #     posterior_mean_switch = avg_S.detach().numpy()
        #     print('estimated posterior mean of feature importance is', posterior_mean_switch)

    print('Finished feature importance Training')

    """ checking the results """
    estimated_params = list(importance.parameters())
    phi_est = F.softplus(torch.Tensor(estimated_params[0]))
    concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
    beta_param = torch.ones(concentration_param.size())
    Gamma_obj = Gamma(concentration_param, beta_param)
    gamma_samps = Gamma_obj.rsample()
    Sstack = gamma_samps / torch.sum(gamma_samps, 0)
    avg_S = torch.mean(Sstack, 1)
    posterior_mean_switch = avg_S.detach().numpy()
    print('estimated posterior mean of feature importance is', posterior_mean_switch)
    posterior_mean[seed_idx,:] = posterior_mean_switch
    print('importance by Q-FIT: ', posterior_mean_switch)

    # store results
    filename = dataset+'posterior_mean.npy'
    np.save(filename, posterior_mean)

""" results """
# after 600 epochs, alpha_0=0.1, mini_batch_size = 2000
# XOR
# [0.47375083 0.47985318 0.00515973 0.00586924 0.00556373 0.00553243
#  0.00638367 0.00603228 0.0059589  0.0058961 ]

# importance by Q-FIT (with alpha_0=0.05) for nonlinear_additive:  [0.30591163 0.08427644 0.06862613 0.09782147 0.08367525 0.0875598
#  0.07724797 0.06898335 0.07020948 0.05568846]