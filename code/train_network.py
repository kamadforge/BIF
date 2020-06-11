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
from models.nn_3hidden import FC, FC_net
from itertools import combinations, chain

import torch.nn as nn
import torch.optim as optim
import torch

mvnrnd = rn.multivariate_normal

import sys
# sys.path.append("/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/data")
from data.tab_dataloader import load_cervical, load_adult, load_credit, load_census, load_isolet, load_adult_short
from data.make_synthetic_datasets import generate_data
from data.make_synthetic_datasets import generate_invase
from data.synthetic_data_loader import synthetic_data_loader





if  __name__ =='__main__':

    """ inputs """
    dataset = "credit" # "xor, orange_skin, or nonlinear_additive"
    method = "nn"
    which_net = 'FC' # FC_net or 'FC'
    rnd_num = 0
    mode = 'training' #training, test&
    prune = True

    rn.seed(rnd_num)

    # try:
    #     x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset)
    #
    # except:
    if 1:

        if dataset == "cervical":
            X_train, y_train, X_test, y_test = load_cervical()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])
        elif dataset == "census":
            X_train, y_train, X_test, y_test = load_census()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])
        elif dataset == "credit":
            X_train, y_train, X_test, y_test = load_credit()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])
        elif dataset == "isolet":
            X_train, y_train, X_test, y_test = load_isolet()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])

        elif dataset == "adult":
            filename = 'adult.p'
            with open(filename, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
                y_tot, x_tot = data
        elif dataset == "adult_short":
            X_train, y_train, X_test, y_test = load_adult_short()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])

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
            x_tot, y_tot, datatypes = generate_data(10000, 'nonlinear_additive')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot}
            np.save('../data/synthetic/nonlinear_additive/dataset_nonlinear_additive.npy', dataset_tosave)

        #the instance depends on 5 features
        elif dataset == "alternating":
            x_tot, y_tot, datatypes = generate_data(10000, 'alternating')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            np.save('../data/synthetic/alternating/dataset_alternating.npy', dataset_tosave)

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
        elif dataset == "total":
            x_tot, y_tot, datatypes = generate_invase(10000, 'total')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            np.save('../data/synthetic/qtip/dataset_total.npy', dataset_tosave)




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
        if which_net == 'FC_net':
            hidden_dim = 400
            which_norm = 'weight_norm'
            model = FC_net(input_num, output_num, hidden_dim, which_norm)
        else:
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


    file_write=True

    # at every repeat, we reshuffle data
    rand_perm_nums = np.random.permutation(N_tot)

    # train and test data
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]
    Xtst = x_tot[rand_perm_nums[N:], :]
    ytst = y_tot[rand_perm_nums[N:]]

    if dataset=="adult_short" or dataset=="credit":
        Xtst=X_test
        X=X_train
        ytst=y_test
        y = y_train

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


    elif mode == "test":

        print(f"dataset: {dataset}")

        if prune:
            print("testing a subset of features")

            features_num=np.arange(Xtst.shape[1])

            def powerset(iterable):
                "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                s = list(iterable)
                return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

            test=Xtst.copy()

            # for result in powerset(features_num):
            #     print(result)
            #
            #     test = Xtst.copy()
            #
            #     if len(result)==4:

            testtype="local"

            if 1:
                if testtype=="global":

                    k = 1

                    met = 2 #2-shap, 3-invase 4-l2x
                    if dataset=="adult":
                        if met==1:
                            important_features=[10,5,0,12,4,7,9,3,8,11,6,1,2,13]#qfit
                        elif met==2:
                            important_features=[7,4,10,0,12,11,6,1,3,5,2,13,9,8]#shap
                        elif met==3:
                            important_features=[5,4,10,0,12,11,9,13,8,3,7,2,1,6]#invase
                        elif met==4:
                            important_features=[5,4,0,9,2,10,11,3,12,1,7,13,6,8]#l2x global
                            important_features=[5,7,10,13,8]#l2x local

                    elif dataset=="credit":
                        if met == 1:
                            important_features = [13,3,11,27,10,9,25,19,6,7,16,8,0,17,5,15,26,21,14,12,4,23,2,28,24,22,1,20,18]  # ,15,6,18,7,12,17] #qfit
                        elif met==2:
                            important_features = [13,16,6,3,11,9,1,7,0,24,4,18,25,5,10,28,22,20,17,8,2,21,26,23,27,14,12,15,19]#,0,19,18,8,9] #shap
                        elif met==3:
                            important_features = [10,13,16,15,7,22,21,17,3,18,2,12,23,0,11,19,24,14,6,4,27,20,8,5,1,26,25,9,28]
                        elif met==4:
                            important_features = [27,19,15,13,21,9,3,26,11,14,6,28,7,18,1,16,2,25,4,5,24,8,17,23,10,22,12,20,0]

                    elif dataset=="cervical":
                        if met==1:
                            important_features = [32,0,11,31,2,1,3,33,4,20,10,15,6,28,9,30,5,17,13,7,8,24,16,23,12,27,14,18,19,22,29,25,26,21]
                        elif met==2:
                            important_features = [32,0,4,5,33,2,3,28,20,10,1,6,31,7,13,30,8,9,11,12,14,27,15,29,17,18,19,21,22,23,24,25,26,16]
                        elif met==3:
                            important_features = [10,32,1,6,5,30,17,33,22,7,12,23,16,21,13,31,9,25,20,19,15,27,18,8,26,14,24,29,4,28,3,11,0,2]
                        elif met==4:
                            important_features = [31,32,25,30,26,3,7,33,11,10,9,8,4,6,5,13,2,1,12,16,14,15,17,18,19,20,21,22,23,24,27,28,29,0]

                    elif dataset=="census":
                        if met== 1:
                            important_features = [17,39,14,5,21,30,10,0,24,16,6,36,18,25,13,26,4,38,9,31,29,7,11,20,27,28,35,19,33,32,12,34,8,22,37,23,3,2,1,15]

                    elif dataset=="isolet":
                        if met==1:
                            important_features = [576,512,466,233,481,396,202,200,472,234,9,265,487,431,545,224,97,425,174]
                        if met==2:
                            important_features = [265,203,13,396,204,533,361,233,459,534,360,9,332]

                    # important_features = result
                    important_features = important_features[:k]
                    unimportant_features = np.delete(features_num, important_features)
                    print("important features: ", important_features, "for met", met)

                    test[:, unimportant_features] = 0




                else: # local test
                    k = 3
                    met = 1  # 2-shap, 3-invase 4-l2x

                    if dataset=="adult_short" and met!=1:
                        dataset="adult"

                    if met == 3:
                        unimportant_features_instance=np.load(f"../comparison_methods/INVASE/INVASE_custom_datasets/instance_featureranks_test_invase_{dataset}_k_{k}.npy")
                    elif met == 4:
                        unimportant_features_instance=np.load(f"../comparison_methods/L2X/instance_featureranks_test_l2x_{dataset}_k_{k}.npy")
                    elif met == 1:
                        unimportant_features_instance = np.load(f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy")


                for i, data in enumerate(test):
                   test[i, unimportant_features_instance[i]]=0



                # no pruning here, just a regular test with all the features present


                i = 0  # choose a sample
                mini_batch_size = 2000
                datatypes_test_samp = None

                LR_model = np.load('models/%s_%s_LR_model0.npy' % (dataset,method), allow_pickle=True)

                model.load_state_dict(LR_model[()][0], strict=False)



                y_pred = model(torch.Tensor(test))
                y_pred = torch.argmax(y_pred, dim=1)

                accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == ytst) / len(ytst))
                print("test accuracy: ", accuracy)


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