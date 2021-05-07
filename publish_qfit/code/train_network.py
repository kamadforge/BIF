"""
1. removed vips data
2.
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
from numpy import genfromtxt



if  __name__ =='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", default="credit") # "xor, orange_skin, or nonlinear_additive"
    parser.add_argument("--mode", default="test") # test, training
    parser.add_argument("--testtype", default="global") #global, local
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
        iter_sigmas = np.array([0.])
        if method == "nn":
            LR_model0 = {}
        for k in range(iter_sigmas.shape[0]):
            sigma = iter_sigmas[k]
            for repeat_idx in range(num_repeat):
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

                LR_model0[repeat_idx] = model.state_dict()

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
                    import json
                    with open('rankings/global_ranks') as f:
                        data = json.load(f)

                    k = 1 #number of important features to keep
                    met = 4 #2-shap, 3-invase 4-l2x, 5-lime

                    path_lime = "../../comparison_methods/LIME/ranks/adult_short_local_ranks.npy"
                    ran = np.load(path_lime)
                    print(ran)

                    rank_str = data[dataset][met]
                    rank = [int(num) for num in rank_str.strip().split(",")]

                    # important_features = result
                    important_features = rank[:k]
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