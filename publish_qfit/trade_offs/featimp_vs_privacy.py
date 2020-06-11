"""
Test learning feature importance under DP and non-DP models
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import argparse
from torch.distributions import Gamma
import pickle
import numpy.random as rn
from Models import Feedforward, Feature_Importance_Model
from sklearn.metrics import roc_auc_score
from Losses import loss_function

from dp_sgd import dp_sgd_backward
from backpack import extend


def main():

    ar = parse()
    rn.seed(ar.seed)

    """ load data """
    filename = 'adult.p'
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, input_dim = x_tot.shape

    use_cuda = not ar.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train and test data
    N = np.int(N_tot * 0.9)
    print('n_data:', N)
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
    optimizer = optim.Adam(classifier.parameters(), lr=ar.lr)

    classifier.train()
    if ar.dp_sigma > 0.:
        extend(classifier)
    # how_many_epochs = 10
    # mini_batch_size = 100
    how_many_iter = np.int(N / ar.clf_batch_size)

    for epoch in range(ar.clf_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            # get the inputs
            inputs = X[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size, :]
            labels = y[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size]

            optimizer.zero_grad()
            y_pred = classifier(torch.Tensor(inputs))
            loss = criterion(y_pred.squeeze(), torch.FloatTensor(labels))

            if ar.dp_sigma > 0.:
                global_norms, global_clips = dp_sgd_backward(classifier.parameters(), loss, device, ar.dp_clip, ar.dp_sigma)
                # print(f'max_norm:{torch.max(global_norms).item()}, mean_norm:{torch.mean(global_norms).item()}')
                # print(f'mean_clip:{torch.mean(global_clips).item()}')
            else:
                loss.backward()
            optimizer.step()
            running_loss += loss.item()

        y_pred = classifier(torch.Tensor(Xtst))
        ROC = roc_auc_score(ytst, y_pred.detach().numpy())
        print('Epoch {}: ROC : {}'.format(epoch, ROC))

    print('Finished Classifier Training')

    if ar.save_model:
        print('saving model')
        torch.save(classifier.state_dict(), f'dp_classifier_sig{ar.dp_sigma}.pt')
        # assert 1 % 1 == 1



#############################################################
    """ learn feature importance """
#############################################################
    seedmax = 6

    mean_importance = np.zeros((seedmax, input_dim))
    phi_est_mat = np.zeros((seedmax, input_dim))

    for seednum in range(1,seedmax):

        rn.seed(seednum)

        """ learn feature importance """
        # input_dim, classifier,
        num_Dir_samps = 1
        importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps)
        # optimizer = optim.Adam(importance.parameters(), lr=0.075)
        optimizer = optim.Adam(importance.parameters(), lr=0.1)

        # We freeze the classifier
        ct = 0
        for child in importance.children():
            ct +=1
            if ct>=1:
                for param in child.parameters():
                    param.requires_grad = False

        # print(list(importance.parameters())) # make sure I only update the gradients of feature importance

        importance.train()
        # how_many_epochs = 100
        # mini_batch_size = 2000  # generally larger mini batch size is more helpful for switch learning
        how_many_iter = np.int(N / ar.switch_batch_size)

        alpha_0 = 0.01
        annealing_rate = 1  # we don't anneal. don't want to think about this.
        kl_term = True

        for epoch in range(ar.switch_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                inputs = X[i * ar.switch_batch_size:(i + 1) * ar.switch_batch_size, :]
                labels = y[i * ar.switch_batch_size:(i + 1) * ar.switch_batch_size]

                optimizer.zero_grad()
                y_pred, phi_cand = importance(torch.Tensor(inputs))
                labels = torch.squeeze(torch.Tensor(labels))
                loss = loss_function(y_pred, labels.view(-1, 1).repeat(1, num_Dir_samps), phi_cand, alpha_0, input_dim,
                                     annealing_rate, N, kl_term)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if np.remainder(epoch,50)==0:
                print('Epoch {}: running_loss : {}'.format(epoch, running_loss/how_many_iter))

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

        mean_importance[seednum,:] = posterior_mean_switch
        phi_est_mat[seednum,:] = phi_est.detach().numpy()

        order_by_importance = np.argsort(posterior_mean_switch)[::-1]
        print('order by importance: ', order_by_importance)

    # [ 7  4 10  0 12 11  6  1  5  3  2  9 13  8]
    # [0:'age', 1:'workclass', 2:'fnlwgt', 3:'education', 4:'education_num',
    #  5:'marital_status', 6:'occupation', 7:'relationship', 8:'race', 9:'sex',
    #  10:'capital_gain', 11:'capital_loss', 12:'hours_per_week', 13:'country'


    filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'importance.npy'
    np.save(filename, mean_importance)

    filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'phi_est.npy'
    np.save(filename, phi_est_mat)

    filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'roc.npy'
    np.save(filename, ROC)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--clf-epochs', type=int, default=20)
    parser.add_argument('--clf-batch-size', type=int, default=1000)

    parser.add_argument('--switch-epochs', type=int, default=400)
    parser.add_argument('--switch-batch-size', type=int, default=20000)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--dp-clip', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=0)

    # for non-private model
    parser.add_argument('--dp-sigma', type=float, default=0.)

    # for private model with varying noise level (privacy level)
    # where each noise level corresponds to these privacy level
    # sig = 1.35 -> eps 8.07 ~= 8
    # sig = 2.3 -> eps 4.01  ~= 4
    # sig = 4.4 -> eps 1.94  ~= 2
    # sig = 8.4 -> eps 0.984 ~= 1
    # sig = 17. -> eps 0.48  ~= 0.5

    # parser.add_argument('--dp-sigma', type=float, default=1.35)
    # parser.add_argument('--dp-sigma', type=float, default=2.3)
    # parser.add_argument('--dp-sigma', type=float, default=4.4)
    # parser.add_argument('--dp-sigma', type=float, default=8.4)
    # parser.add_argument('--dp-sigma', type=float, default=17.)

    return parser.parse_args()


if __name__ == '__main__':
    main()
