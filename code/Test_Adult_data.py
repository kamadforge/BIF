"""
This script is written based on what I wrote back on July 2, 2019 for testing VIPS on Adult data
"""

__author__ = 'mijung'

import Bayesian_Logistic_Regression as VIPS_BLR_MA # this has all core functions
import os
import sys
import scipy
import scipy.io
import numpy as np
import numpy.random as rn
from sklearn.metrics import roc_curve,auc
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import autodp
from autodp import privacy_calibrator
from models.nn_3hidden import FC

import torch.nn as nn
import torch.optim as optim
import torch

mvnrnd = rn.multivariate_normal

import sys
# sys.path.append("/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/data")
from data.tab_dataloader import load_cervical, load_adult, load_credit
from data.make_synthetic_datasets import generate_data
from data.make_synthetic_datasets import generate_invase


if  __name__ =='__main__':

    """ inputs """
    dataset = "syn5" # "xor, orange_skin, or nonlinear_additive"
    method = "nn"
    rnd_num = 0

    rn.seed(rnd_num)

    if dataset == "cervical":
        X_train, y_train, X_test, y_test = load_cervical()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])
    elif dataset == "credit":
        X_train, y_train, X_test, y_test = load_credit()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])
    elif dataset == "adult":
        filename = 'adult.p'
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            y_tot, x_tot = data
    elif dataset=="xor":
        x_tot, y_tot, datatypes = generate_data(10000, 'XOR')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_XOR={'x': x_tot, 'y':y_tot}
        np.save('../data/synthetic/XOR/dataset_XOR.npy', dataset_XOR)
    elif dataset == "orange_skin":
        x_tot, y_tot, datatypes = generate_data(10000, 'orange_skin')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot}
        np.save('../data/synthetic/orange_skin/dataset_orange_skin.npy', dataset_tosave)
    elif dataset == "nonlinear_additive":
        x_tot, y_tot, datatypes = generate_data(10000, 'orange_skin')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot}
        np.save('../data/synthetic/nonlinear_additive/dataset_nonlinear_additive.npy', dataset_tosave)

    #the instance depends on 5 features
    elif dataset == "alternating":
        x_tot, y_tot, datatypes = generate_data(10000, 'alternating')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
        #np.save('../data/synthetic/alternating/dataset_alternating.npy', dataset_tosave)

    #the instant depends only on 1 feature, all other features for all the instances in the dataset are either 1 or 0
    elif dataset == "syn4":
        x_tot, y_tot, datatypes = generate_invase(10000, 'syn4')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
        np.save('../data/synthetic/invase/dataset_syn4.npy', dataset_tosave)
    elif dataset == "syn5":
        x_tot, y_tot, datatypes = generate_invase(10000, 'syn5')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
        np.save('../data/synthetic/invase/dataset_syn5.npy', dataset_tosave)
    elif dataset == "syn6":
        x_tot, y_tot, datatypes = generate_invase(10000, 'syn6')
        y_tot = np.argmax(y_tot, axis=1)
        dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
        np.save('../data/synthetic/invase/dataset_syn6.npy', dataset_tosave)




    ####################################
    # define essential quantities
    output_num = 2
    sample_num, input_num = x_tot.shape

    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)
    N_test = N_tot - N

    if method == "vips":
        """ hyper-params for the prior over the parameters """
        alpha = 0.02
        a0 = 1.
        b0 = 1.

        """ stochastic version """
        tau0 = 1024
        kappa = 0.7
        MaxIter = 200 # EM iteration
        nu = 0.04
        S = np.int(nu*N)
        print('mini batch size is ', S)

        exp_nat_params_prv = np.ones([d,d])
        mean_alpha_prv = a0/b0

    elif method == "nn":
        model = FC(input_num, output_num)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        num_epochs = 500

    """ set the privacy parameter """
    # dp_epsilon = 1
    # dp_delta = 1/N_tot
    # k = MaxIter*2 # two expected suff stats
    # params = privacy_calibrator.gaussian_mech(dp_epsilon, dp_delta, prob=nu, k=k)
    # sigma = params['sigma']
    # print('privacy parameter is ', sigma)
    # iter_sigmas = np.array([0, sigma])  # test non-private first, then private with the desired epsilon level

    mode='training'
    file_write=True

    if mode=='training':

        num_repeat = 1

        # iter_sigmas = np.array([0., 1., 10., 50., 100.]) # use this one when we have different levels of noise.
        iter_sigmas = np.array([0.])

        if method == "vips":
            auc_private_stoch_ours = np.empty([iter_sigmas.shape[0], num_repeat])
            LR_model0 = np.empty([num_repeat, d])
            LR_model1 = np.empty([num_repeat, d])
            LR_model10 = np.empty([num_repeat, d])
            LR_model50 = np.empty([num_repeat, d])
            LR_model100 = np.empty([num_repeat, d])
        elif method == "nn":
            LR_model0 = {}

        for k in range(iter_sigmas.shape[0]):
            sigma = iter_sigmas[k]

            for repeat_idx in range(num_repeat):

                # at every repeat, we reshuffle data
                rand_perm_nums = np.random.permutation(N_tot)

                #train and test data
                X = x_tot[rand_perm_nums[0:N], :]
                y = y_tot[rand_perm_nums[0:N]]
                Xtst = x_tot[rand_perm_nums[N:], :]
                ytst = y_tot[rand_perm_nums[N:]]

                if method=="vips":
                    for iter in range(MaxIter):

                        # VI iterations start here
                        rhot = (tau0+iter)**(-kappa)

                        exp_suff_stats1, exp_suff_stats2 = VIPS_BLR_MA.VBEstep_private(sigma, X, y, exp_nat_params_prv)

                        if iter==0:
                            nu_old = []
                            ab_old = []
                        nu_new, ab_new, exp_nat_params, mean_alpha, Mu_theta = VIPS_BLR_MA.VBMstep_stochastic(rhot, nu_old, ab_old, N, a0, b0, exp_suff_stats1, exp_suff_stats2, mean_alpha_prv, iter)

                        mean_alpha_prv = mean_alpha
                        exp_nat_params_prv = exp_nat_params
                        nu_old = nu_new
                        ab_old = ab_new

                        """ compute roc_curve and auc """
                        ypred = VIPS_BLR_MA.computeOdds(Xtst, Mu_theta)
                        accuracy = (np.sum(np.round(ypred) == ytst) / len(ytst))
                        print("iter number: ", iter)
                        print("test accuracy: ", accuracy)

                elif method=="nn":

                    for epoch in range(num_epochs):

                        print("epoch number: ", epoch)

                        optimizer.zero_grad()
                        ypred_tr = model(torch.Tensor(X))
                        loss = criterion(ypred_tr, torch.LongTensor(y))
                        loss.backward()
                        optimizer.step()

                        # print(loss)

                        ###########
                        # TEST per epoch

                        y_pred = model(torch.Tensor(Xtst))
                        y_pred = torch.argmax(y_pred, dim=1)

                        accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == ytst) / len(ytst))
                        print("test accuracy: ", accuracy)


                if method=="vips":
                    # last model is saved in every sigma value
                    if k==0:
                        LR_model0[repeat_idx,:] = Mu_theta
                    elif k==1:
                        LR_model1[repeat_idx,:] = Mu_theta
                    elif k==2:
                        LR_model10[repeat_idx,:] = Mu_theta
                    elif k==3:
                        LR_model50[repeat_idx, :] = Mu_theta
                    else: #k==4
                        LR_model100[repeat_idx,:] = Mu_theta

                elif method=="nn":
                    LR_model0[repeat_idx] = model.state_dict()

        if method == "vips":
            # np.save('models/%s_accuracy_ours' % (dataset, method), auc_private_stoch_ours)
            np.save('models/%s_%s_LR_model0' % (dataset, method), LR_model0)
            np.save('models/%s_%s_LR_model1' % (dataset, method), LR_model1)
            np.save('models/%s_%s_LR_model10' % (dataset, method), LR_model10)
            np.save('models/%s_%s_LR_model50' % (dataset, method), LR_model50)
            np.save('models/%s_%s_LR_model100' % (dataset, method), LR_model100)
        elif method == "nn":
            np.save('models/%s_%s_LR_model0' % (dataset, method), LR_model0)


    # elif mode=='test':
    #
    #     model=np.load('LR_model0.npy') #number of runs x number of features, e.g. [20,14]
    #
    #     print(model.shape)
    #
    #     rand_perm_nums = np.random.permutation(N_tot)
    #
    #     # test data
    #     Xtst = x_tot[rand_perm_nums[N:], :]
    #     ytst = y_tot[rand_perm_nums[N:]]
    #
    #
    #     from itertools import combinations
    #
    #     s = np.arange(0, model.shape[1])  # list from 0 to 19 as these are the indices of the data tensor
    #     for r in range(1, model.shape[1]):  # produces the combinations of the elements in s
    #         results = []
    #         for combination in list(combinations(s, r)):
    #             #combination = torch.LongTensor(combination)
    #
    #             print("\n")
    #             print(combination)
    #
    #             model_to_test= model[0]
    #             print(model_to_test)
    #
    #             model_to_test_pruned = model_to_test.copy()
    #             model_to_test_pruned[np.array(combination)]=0
    #             print(model_to_test_pruned)
    #             Xtst[:, np.array(combination)] = 0
    #
    #             ypred = VIPS_BLR_MA.computeOdds(Xtst, model_to_test_pruned)  # model[0] is one run
    #
    #             accuracy = (np.sum(np.round(ypred.flatten()) == ytst) / len(ytst))
    #             print("Accuracy: ", accuracy)
    #
    #             if file_write:
    #                 with open("combinations/model[0].txt", "a+") as textfile:
    #                     textfile.write("%s: %.4f\n" % (",".join(str(x) for x in combination), accuracy))