"""
1. removed vips data
"""

__author__ = 'mijung'

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
#from autodp import privacy_calibrator
from models.nn_3hidden import FC, FC_net
from itertools import combinations, chain
import argparse
from sklearn.metrics import matthews_corrcoef
import torch.nn as nn
import torch.optim as optim
import torch
mvnrnd = rn.multivariate_normal
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).resolve().parent / "data"))
import sys
from tab_dataloader import load_adult, load_credit, load_adult_short
from tab_dataloader import load_intrusion, load_covtype
from make_synthetic_datasets import generate_data
from make_synthetic_datasets import generate_invase
#from data.synthetic_data_loader import synthetic_data_loader


if  __name__ =='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xor") # "xor, orange_skin, or nonlinear_additive"
    parser.add_argument("--mode", default="test") # test, training
    parser.add_argument("--testtype", default="local") #global, local
    parser.add_argument("--prune", default=True) #tests the subset of features
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--met", default=1, type=int) #1-qfit,  2-shap, 3-invase 4-l2x
    args=parser.parse_args()
    dataset = args.dataset
    method = "nn"
    which_net = 'FC' # FC_net or 'FC'
    rnd_num = 0
    mode = args.mode
    prune = args.prune
    rn.seed(rnd_num)

    def save_dataset(path, dataset):
        if not os.path.isdir(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        np.save(path, dataset)
    if 1:
        if dataset == "credit":
            X_train, y_train, X_test, y_test = load_credit()
            x_tot = np.concatenate([X_train, X_test])
            y_tot = np.concatenate([y_train, y_test])
        if dataset == "intrusion":
            X_train, y_train, X_test, y_test = load_intrusion()
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
            save_dataset('../data/synthetic/XOR/dataset_XOR.npy', dataset_XOR)
        elif dataset == "orange_skin":
            x_tot, y_tot, datatypes = generate_data(10000, 'orange_skin')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot}
            save_dataset('../data/synthetic/orange_skin/dataset_orange_skin.npy', dataset_tosave)
        elif dataset == "nonlinear_additive":
            x_tot, y_tot, datatypes = generate_data(10000, 'nonlinear_additive')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot}
            save_dataset('../data/synthetic/nonlinear_additive/dataset_nonlinear_additive.npy', dataset_tosave)
        #the instance depends on 5 features
        elif dataset == "alternating":
            x_tot, y_tot, datatypes = generate_data(10000, 'alternating')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/alternating/dataset_alternating.npy', dataset_tosave)
        #the instant depends only on 1 feature, all other features for all the instances in the dataset are either 1 or 0
        elif dataset == "syn4":
            x_tot, y_tot, datatypes = generate_invase(10000, 'syn4')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/invase/dataset_syn4.npy', dataset_tosave)
        elif dataset == "syn5":
            x_tot, y_tot, datatypes = generate_invase(10000, 'syn5')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/invase/dataset_syn5.npy', dataset_tosave)
        elif dataset == "syn6":
            x_tot, y_tot, datatypes = generate_invase(10000, 'syn6')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/invase/dataset_syn6.npy', dataset_tosave)
        elif dataset == "total":
            x_tot, y_tot, datatypes = generate_invase(10000, 'total')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/qtip/dataset_total.npy', dataset_tosave)

    ####################################
    # define essential quantities
    if dataset=="intrusion":
        output_num = 4
    else:
        output_num = 2
    sample_num, input_num = x_tot.shape
    N_tot, d = x_tot.shape
    training_data_por = 0.8
    N = int(training_data_por * N_tot)
    N_test = N_tot - N


    if method == "nn":
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


    if dataset=="adult_short" or dataset=="credit" or dataset=="intrusion":
        Xtst=X_test
        X=X_train
        ytst=y_test
        y = y_train


    if mode=='train':

        num_repeat = 1
        # iter_sigmas = np.array([0., 1., 10., 50., 100.]) # use this one when we have different levels of noise.
        iter_sigmas = np.array([0.])
        if method == "nn":
            LR_model0 = {}
        for k in range(iter_sigmas.shape[0]):
            sigma = iter_sigmas[k]

            for repeat_idx in range(num_repeat):

                if method=="nn":

                    for epoch in range(num_epochs):
                        print("epoch number: ", epoch)
                        optimizer.zero_grad()
                        ypred_tr = model(torch.Tensor(X))
                        loss = criterion(ypred_tr, torch.LongTensor(y))
                        loss.backward()
                        optimizer.step()

                        ###########
                        # TEST per epoch

                        y_pred = model(torch.Tensor(Xtst))
                        y_pred = torch.argmax(y_pred, dim=1)
                        accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == ytst) / len(ytst))
                        print("test accuracy: ", accuracy)

                if method=="nn":
                    LR_model0[repeat_idx] = model.state_dict()

        if method == "nn":
            if not os.path.isdir("checkpoints"):
                os.mkdir("checkpoints")
            np.save('checkpoints/%s_%s_LR_model0_epochs_%d_acc_%.2f' % (dataset, method, num_epochs, accuracy), LR_model0)

    ###########################################

    elif mode == "test":

        print(f"dataset: {dataset}")

        # pruning
        if prune:
            print("testing a subset of features")
            features_num=np.arange(Xtst.shape[1])
            #experimental feature
            def powerset(iterable):
                "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
                s = list(iterable)
                return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
            test=Xtst.copy()

            if 1:
                if args.testtype=="global":
                    k = 1 #number of important features to keep
                    met = 1 #2-shap, 3-invase 4-l2x
                    if dataset=="adult":
                        if met==1:
                            important_features=[10,5,0,12,4,7,9,3,8,11,6,1,2,13]#qfit
                        elif met==2:
                            important_features=[7,4,10,0,12,11,6,1,3,5,2,13,9,8]#shap
                        elif met==3:
                            important_features=[5,4,10,0,12,11,9,13,8,3,7,2,1,6]#invase
                        elif met==4:
                            important_features=[5,4,0,9,2,10,11,3,12,1,7,13,6,8]#l2x global

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

                    elif dataset=="intrusion": #dummy rankings
                        if met==1:
                            important_features = [26, 17, 27, 18,  5,  0, 36,  2,  8, 29, 35,  3, 21,  1,  6, 30,  9, 15, 16, 32, 20, 28, 22, 11, 39, 19, 24, 31,  7, 13, 10, 25, 23, 14, 12, 37, 34, 38, 33,  4]
                            important_features = [17, 27, 26, 18,  5,  0, 14, 12, 23, 33, 32,  2,  1, 11, 38,  8, 29, 37, 3, 21, 16, 39, 20,  9, 19, 25, 10, 13, 15, 24, 35, 31, 34, 22, 28,  7, 30,  6, 36,  4]
                            important_features = [17, 27, 18,  5, 26,  0,  4, 28,  8, 39, 25, 11, 33,  2,  6,  9, 13, 21,
        10, 36, 19, 12, 24,  1, 16, 35, 32, 31, 23, 38, 22, 15, 29, 30, 20,  7,
         3, 37, 14, 34]
                        elif met==2:
                            important_features = [17,5,32,24,30,2,4,3,28,8,1,10,29,27,6,33,37,26,0,23,18,21,9,13,35,15,31,36,7,39,11,12,14,16,38,20,22,25,34,19]
                        elif met==3:
                            important_features = [33,17,30,2,27,13,9,15,7,35,16,3,25,12,21,28,39,8,18,29,14,20,38,24,10,23,6,31,0,1,19,11,36,34,4,32,5,37,22,26]
                        elif met==4:
                            important_features = [24, 29, 31, 34, 18, 16, 11, 35, 32,  1, 39, 38, 36,  6, 23,  5, 19, 15, 27,  0, 37, 12, 13  ,2,14, 10, 22, 20 ,33 ,26,  9,  8, 25, 17, 21,  7, 30,  4,  3, 28] #l2x

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
                    #pruning global
                    print("pruning global")
                    test[:, unimportant_features] = 0

                ###################
                # local test
                else:
                    k = args.k
                    met = args.met  #1-qfit,  2-shap, 3-invase 4-l2x
                    met_names = {1 : "qfit", 2: "shap", 3: "invase", 4: "l2x", 5: "shap"}

                    if dataset=="adult_short" and met!=1:
                        dataset="adult"

                    if met == 1:
                        dir_ranks = "publish_qfit/code/rankings"
                    elif met == 2:
                        dir_ranks = "comparison_methods/SHAP/ranks"
                    elif met == 3:
                        dir_ranks  = "comparison_methods/INVASE/INVASE_custom_datasets/ranks"
                    elif met == 4:
                        dir_ranks = "comparison_methods/L2X/ranks"



                    for file in os.listdir(os.path.join("../../", dir_ranks)):
                        if dataset in file and f"k_{k}" in file:
                            unimportant_features_instance = np.load(os.path.join("../../", dir_ranks, file))
                            print(f"loaded dataset '{met_names[met]}' in '{file}' from '{dir_ranks}'")

                    if met == 2:
                        features_rank = np.load(os.path.join("../../", dir_ranks, "shap_"+dataset+".npy"))
                    unimportant_features_instance = features_rank[:, k:]

                    #pruning local
                    print(f"unimportant features shape: {unimportant_features_instance.shape}")
                    print("pruning local")
                    for i, data in enumerate(test):
                        test[i, unimportant_features_instance[i]] = 0

                ###########################
                # pruning (both local and global)
                # zeroing values in the input dataset
                print(f"the shape of test dataset is: {test.shape}")


                i = 0  # choose a sample
                mini_batch_size = 2000
                datatypes_test_samp = None

                # loading the trained model on a dataset (credit, adult, intrusion, etc.)
                for file in os.listdir("checkpoints"):
                    if dataset in file:
                        LR_model = np.load(os.path.join("checkpoints", file), allow_pickle=True)
                        print("Loaded: ", file)
                model.load_state_dict(LR_model[()][0], strict=False)

                # testing the subset of features on a trained model
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