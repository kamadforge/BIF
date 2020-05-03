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

import sys
sys.path.append("/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/data")
from tab_dataloader import load_cervical, load_adult, load_credit

if  __name__ =='__main__':

    dataset = 'credit'

    """ inputs """
    rnd_num = 123
    rn.seed(rnd_num)

    if dataset == "cervical":
        X_train, y_train, X_test, y_test = load_cervical()
    elif dataset == "credit":
        X_train, y_train, X_test, y_test = load_credit()
    elif dataset == "adult":
        filename = 'adult.p'
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

    x_tot = np.concatenate([X_train, X_test])
    y_tot = np.concatenate([y_train, y_test])

    # unpack data
    #y_tot, x_tot = data
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
    MaxIter = 20 # EM iteration
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

    mode='training'
    file_write=True

    if mode=='training':

        num_repeat = 20

        iter_sigmas = np.array([0., 1., 10., 50., 100.])
        auc_private_stoch_ours = np.empty([iter_sigmas.shape[0], num_repeat])
        LR_model0 = np.empty([num_repeat, d])
        LR_model1 = np.empty([num_repeat, d])
        LR_model10 = np.empty([num_repeat, d])
        LR_model50 = np.empty([num_repeat, d])
        LR_model100 = np.empty([num_repeat, d])

        for k in range(iter_sigmas.shape[0]):
            sigma = iter_sigmas[k]

            for repeat_idx in range(num_repeat):

                # at every repeat, we reshuffle data
                rand_perm_nums = np.random.permutation(N_tot)

                #train and test data
                # X = x_tot[rand_perm_nums[0:N], :]
                # y = y_tot[rand_perm_nums[0:N]]
                # Xtst = x_tot[rand_perm_nums[N:], :]
                # ytst = y_tot[rand_perm_nums[N:]]

                X= X_train
                y=y_train
                Xtst = X_test
                ytst = y_test


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

                    # Mu_theta=[7.19253774,
                    # 16.59639552 ,- 1.56681484, - 3.04541022, - 4.80905384, - 8.10952018,
                    # 10.40246059, - 2.60223157, - 8.53590502,
                    # 6.96936155,
                    # 2.43518547,
                    # 6.89259258,
                    # 22.69197287, - 6.51335088]

     #                ypred [2.42630590e-22 7.89200116e-22 1.00000000e+00 ... 8.02175969e-04
     # 1.00000000e+00 1.00000000e+00]

                    #0.7002349098247271

                    """ compute roc_curve and auc """
                    ypred = VIPS_BLR_MA.computeOdds(Xtst, Mu_theta)
                    #print(Xtst[1:3])
                    # [[-0.55583011  0.09005041  0.08217687 - 0.33543693  1.13473876  0.92163395
                    #   - 1.3178091   0.96694656  0.39366753 - 1.42233076 - 0.14592048 - 0.21665953
                    #   - 0.03542945  0.29156857]
                    #  [-0.18926719  0.09005041  0.11360322 - 2.40251115 - 1.19745882  0.92163395
                    #  1.04693249 - 0.27780504 - 1.96262077  0.70307135  1.6888354 - 0.21665953
                    #  - 0.03542945  0.29156857]]
                    #print("ypred", ypred)


                    fal_pos_rate_tst, true_pos_rate_tst, thrsld_tst = roc_curve(ytst, ypred.flatten())
                    auc_tst = auc(fal_pos_rate_tst,true_pos_rate_tst)


                    print("Mu", Mu_theta)
                    print("auc", auc_tst)
                    # update iteration number
                    iter = iter + 1

                print('AUC is', auc_tst)
                print('sigma is', sigma)

                accuracy = (np.sum(np.round(ypred.flatten()) == ytst) / len(ytst))
                print("Accuracy: ", accuracy)

                auc_private_stoch_ours[k, repeat_idx] = auc_tst

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

        np.save('models/%s_accuracy_ours' % dataset, auc_private_stoch_ours)
        np.save('models/%s_LR_model0' % dataset, LR_model0)
        np.save('models/%s_LR_model1' % dataset, LR_model1)
        np.save('models/%s_LR_model10' % dataset, LR_model10)
        np.save('models/%s_LR_model50' % dataset, LR_model50)
        np.save('models/%s_LR_model100' % dataset, LR_model100)

    elif mode=='test':

        model=np.load('LR_model0.npy') #number of runs x number of features, e.g. [20,14]

        print(model.shape)

        rand_perm_nums = np.random.permutation(N_tot)

        # test data
        Xtst = x_tot[rand_perm_nums[N:], :]
        ytst = y_tot[rand_perm_nums[N:]]


        from itertools import combinations

        s = np.arange(0, model.shape[1])  # list from 0 to 19 as these are the indices of the data tensor
        for r in range(1, model.shape[1]):  # produces the combinations of the elements in s
            results = []
            for combination in list(combinations(s, r)):
                #combination = torch.LongTensor(combination)

                print("\n")
                print(combination)

                model_to_test= model[0]
                print(model_to_test)

                model_to_test_pruned = model_to_test.copy()
                model_to_test_pruned[np.array(combination)]=0
                print(model_to_test_pruned)
                Xtst[:, np.array(combination)] = 0

                ypred = VIPS_BLR_MA.computeOdds(Xtst, model_to_test_pruned)  # model[0] is one run

                accuracy = (np.sum(np.round(ypred.flatten()) == ytst) / len(ytst))
                print("Accuracy: ", accuracy)

                if file_write:
                    with open("combinations/model[0].txt", "a+") as textfile:
                        textfile.write("%s: %.4f\n" % (",".join(str(x) for x in combination), accuracy))