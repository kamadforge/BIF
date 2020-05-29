"""
Test learning global feature importance with/without switch net for Syn1-3 datasets.

"""

__author__ = 'mijung'
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
from Mijungs_Models import Feedforward, Feature_Importance_Model
from Mijungs_Losses import loss_function

np.random.seed(4)

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
classifier = Feedforward(input_dim, 100, 50)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.005)

classifier.train()
epoch = 100

for epoch in range(epoch):

    optimizer.zero_grad()
    y_pred = classifier(torch.Tensor(X))
    loss = criterion(y_pred.squeeze(), torch.FloatTensor(y))
    loss.backward()
    optimizer.step()

    y_pred = classifier(torch.Tensor(Xtst))

    accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == ytst) / len(ytst))
    print('Epoch {}: test accuracy: {}'.format(epoch, accuracy))


""" learn feature importance """
# input_dim, classifier,
num_Dir_samps = 1
importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps)
optimizer = optim.Adam(importance.parameters(), lr=0.075)

# We freeze the classifier
ct = 0
for child in importance.children():
    ct +=1
    if ct>=1:
        for param in child.parameters():
            param.requires_grad = False

# print(list(importance.parameters())) # make sure I only update the gradients of feature importance

importance.train()
epoch = 400
alpha_0 = 0.1
annealing_rate = 1 # we don't anneal. don't want to think about this.
kl_term = True

for epoch in range(epoch):

    optimizer.zero_grad()
    y_pred, phi_cand = importance(torch.Tensor(X))
    labels = torch.squeeze(torch.Tensor(y))
    loss = loss_function(y_pred, labels.view(-1, 1).repeat(1, num_Dir_samps), phi_cand, alpha_0, input_dim, annealing_rate, N, kl_term)
    loss.backward()
    optimizer.step()

    print('Epoch {}: training loss: {}'.format(epoch, loss))

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