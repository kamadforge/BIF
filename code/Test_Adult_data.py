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

mvnrnd = rn.multivariate_normal

if  __name__ =='__main__':

    """ inputs """
    rnd_num = 123
    rn.seed(rnd_num)

    """ load data """
    filename = 'adult.p'

    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)
    N_test = N_tot - N

    """ hyper-params for the prior over the parameters """
    alpha = 0.02
    a0 = 1.
    b0 = 1.

    """ stochastic version """
    tau0 = 1024
    kappa = 0.7
    MaxIter = 200 # EM iteration
    nu = 0.005
    S =  np.int(nu*N)
    print('mini batch size is ', S)

    exp_nat_params_prv = np.ones([d,d])
    mean_alpha_prv = a0/b0

    """ set the privacy parameter """
    # dp_epsilon = 1
    # dp_delta = 1/N_tot
    # k = MaxIter*2 # two expected suff stats
    # params = privacy_calibrator.gaussian_mech(dp_epsilon, dp_delta, prob=nu, k=k)
    # sigma = params['sigma']
    # print('privacy parameter is ', sigma)
    # iter_sigmas = np.array([0, sigma])  # test non-private first, then private with the desired epsilon level

    num_repeat = 20

    iter_sigmas = np.array([0., 1., 10., 50., 100.])
    auc_private_stoch_ours = np.empty([iter_sigmas.shape[0], num_repeat])
    LR_models = np.empty([iter_sigmas.shape[0], d])

    for k in range(iter_sigmas.shape[0]):
        sigma = iter_sigmas[k]

        for repeat_idx in range(num_repeat):

            # at every repeat, we reshuffle data
            rand_perm_nums = np.random.permutation(N_tot)

            X = x_tot[rand_perm_nums[0:N], :]
            y = y_tot[rand_perm_nums[0:N]]
            Xtst = x_tot[rand_perm_nums[N:], :]
            ytst = y_tot[rand_perm_nums[N:]]


            for iter in range(MaxIter):

                # VI iterations start here

                rhot = (tau0+iter)**(-kappa)

                """ select a new mini-batch """
                rand_perm_nums =  np.random.permutation(N)
                idx_minibatch = rand_perm_nums[0:S]
                xtrain_m = X[idx_minibatch,:]
                ytrain_m = y[idx_minibatch]

                exp_suff_stats1, exp_suff_stats2 = VIPS_BLR_MA.VBEstep_private(sigma, xtrain_m, ytrain_m, exp_nat_params_prv)

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
                fal_pos_rate_tst, true_pos_rate_tst, thrsld_tst = roc_curve(ytst, ypred.flatten())
                auc_tst = auc(fal_pos_rate_tst,true_pos_rate_tst)

                # update iteration number
                iter = iter + 1

            print('AUC is', auc_tst)
            print('sigma is', sigma)

            auc_private_stoch_ours[k, repeat_idx] = auc_tst

        # last model is saved in every sigma value
        LR_models[k,:] = Mu_theta

    np.save('accuracy_ours', auc_private_stoch_ours)
    np.save('LR_models', LR_models)