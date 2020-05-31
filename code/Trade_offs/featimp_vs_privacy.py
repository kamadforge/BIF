"""
Test learning feature importance under DP and non-DP models
"""
__author__ = 'mijung'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import pickle
import numpy.random as rn
from Models import Feedforward, Feature_Importance_Model
from sklearn.metrics import roc_auc_score
from Losses import loss_function

def main():

    rnd_num = 0
    rn.seed(rnd_num)

    """ load data """
    filename = 'adult.p'
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, input_dim = x_tot.shape

    """ set the privacy parameter """
    # dp_epsilon = 1
    # dp_delta = 1/N_tot
    # k = ? , # state the number of iterations
    # params = privacy_calibrator.gaussian_mech(dp_epsilon, dp_delta, prob=nu, k=k)
    # sigma = params['sigma']
    # print('privacy parameter is ', sigma)

    # train and test data
    N = np.int(N_tot * 0.9)
    rand_perm_nums = np.random.permutation(N_tot)
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]
    Xtst = x_tot[rand_perm_nums[N:], :]
    ytst = y_tot[rand_perm_nums[N:]]

#############################################################
    """ train a classifier """
#############################################################
    classifier = Feedforward(input_dim, 100, 20)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.05)

    classifier.train()
    how_many_epochs = 10
    mini_batch_size = 100
    how_many_iter = np.int(N / mini_batch_size)

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            # get the inputs
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


#############################################################
    """ learn feature importance """
#############################################################

    """ learn feature importance """
    # input_dim, classifier,
    num_Dir_samps = 10
    importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps)
    optimizer = optim.Adam(importance.parameters(), lr=0.005)

    # We freeze the classifier
    ct = 0
    for child in importance.children():
        ct +=1
        if ct>=1:
            for param in child.parameters():
                param.requires_grad = False

    # print(list(importance.parameters())) # make sure I only update the gradients of feature importance

    importance.train()
    how_many_epochs = 100
    mini_batch_size = 2000 # generally larger mini batch size is more helpful for switch learning
    how_many_iter = np.int(N / mini_batch_size)

    alpha_0 = 0.1
    annealing_rate = 1 # we don't anneal. don't want to think about this.
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

        print('Epoch {}: running_loss : {}'.format(epoch, running_loss/how_many_iter))

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

    print('Finished feature importance Training')





if __name__ == '__main__':
    main()