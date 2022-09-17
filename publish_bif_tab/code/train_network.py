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
import json



if  __name__ =='__main__':

    ###############
    # GET ARGS

    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xor", choices=["xor", "orange_skin", "nonlinear_additive", "syn4", "syn5", "syn6", "credit", "adult_short", "intrusion"])
    parser.add_argument("--mode", default="train") # test, train
    parser.add_argument("--testtype", default="local") #global, local
    parser.add_argument("--prune", default=True) #tests the subset of features
    parser.add_argument("--ktop", default=5, type=int)
    parser.add_argument("--met", default=4, type=int) #0-bif,  1-shap, 2-invase 3-l2x 4-lime
    parser.add_argument("--train_epochs", default=50, type=int) #500
    args=parser.parse_args()
    dataset = args.dataset
    method = "nn"
    which_net = 'FC' # FC_net or 'FC'
    rnd_num = 0
    mode = args.mode
    prune = args.prune
    rn.seed(rnd_num)

    #######################
    # GET DATA

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
        elif dataset == "xor_mean5":
            x_tot, y_tot, datatypes = generate_data(10000, 'XOR_mean5')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_XOR = {'x': x_tot, 'y': y_tot}
            save_dataset('../data/synthetic/XOR/dataset_XOR_mean5.npy', dataset_XOR)
        elif dataset == "orange_skin":
            x_tot, y_tot, datatypes = generate_data(10000, 'orange_skin')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot}
            save_dataset('../data/synthetic/orange_skin/dataset_orange_skin.npy', dataset_tosave)
        elif dataset == "orange_skin_mean5":
            x_tot, y_tot, datatypes = generate_data(10000, 'orange_skin_mean5')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot}
            save_dataset('../data/synthetic/orange_skin/dataset_orange_skin_mean5.npy', dataset_tosave)
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
        elif dataset == "syn4_mean5":
            x_tot, y_tot, datatypes = generate_invase(10000, 'syn4')
            y_tot = np.argmax(y_tot, axis=1)
            dataset_tosave = {'x': x_tot, 'y': y_tot, 'datatypes': datatypes}
            save_dataset('../data/synthetic/invase/dataset_syn4_mean5.npy', dataset_tosave)
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
        num_epochs = args.train_epochs

    file_write=True

    # at every repeat, we reshuffle data
    rand_perm_nums = np.random.permutation(N_tot)

    # train and test data
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]
    Xtst = x_tot[rand_perm_nums[N:], :]
    ytst = y_tot[rand_perm_nums[N:]]
    global Xtst_means
    Xtst_means = np.mean(Xtst, 0)
    if "syn" in dataset:
        datatypes_tr = datatypes[rand_perm_nums[0:N]]
        datatypes_tst=datatypes[rand_perm_nums[N:]]
        datatypes_tst_num_relevantfeatures = np.sum(datatypes_tst, axis=1)

    if dataset=="adult_short" or dataset=="credit" or dataset=="intrusion":
        Xtst=X_test
        X=X_train
        ytst=y_test
        y = y_train

    #################
    # TRAIN

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
    # TEST FOR SYNTHETIC DATASETS (GLOBAL AND LOCAL) (TABLE 1)

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

            ktop = args.ktop
            met = args.met  # 1-bif,  2-shap, 3-invase 4-l2x
            met_names = {1: "bif", 2: "shap", 3: "invase", 4: "l2x", 5: "shap"}

            if 1:
                if args.testtype=="global": #global test
                    global_json_path = 'rankings/global_ranks'
                    print(f"The ranks from the JSON file in {global_json_path}")
                    with open(global_json_path) as f:
                        data = json.load(f)
                    try:
                        rank_str = data[dataset][met]
                    except KeyError:
                        print("\nError: The dataset is not suited for global selection")
                        exit()
                    rank = [int(num) for num in rank_str.strip().split(",")]
                    # important_features = result
                    important_features = rank[:ktop]
                    unimportant_features = np.delete(features_num, important_features)
                    print("important features: ", important_features, "for met", met)
                    #pruning global
                    print("pruning global")
                    test[:, unimportant_features] = Xtst_means[unimportant_features]
                    #test[:, unimportant_features] = 0

                else: # local test
                    k = args.ktop
                    met = args.met  #1-bif,  2-shap, 3-invase 4-l2x
                    met_names = {0 : "bif", 1: "shap", 2: "invase", 3: "l2x", 4: "lime"}
                    print("Method: ", met_names[met])

                    if dataset=="adult_short" and met!=1 and met!=4:
                        dataset="adult"

                    if met == 0:
                        dir_ranks = "publish_bif_tab/code/rankings"
                    elif met == 1:
                        dir_ranks = "comparison_methods/SHAP/ranks"
                    elif met == 2:
                        dir_ranks  = "comparison_methods/INVASE/INVASE_custom_datasets/ranks"
                    elif met == 3:
                        dir_ranks = "comparison_methods/L2X/ranks"
                    elif met == 4:
                        dir_ranks = "comparison_methods/LIME/ranks"

                    if (met == 2 or met ==3):
                        file = f"instance_featureranks_test_{met_names[met]}_{dataset}_k_{k}.npy"
                    if met == 4:
                        file = f"{dataset}_local_ranks.npy"
                    if met == 1:
                        file = "shap_"+dataset+".npy"

                    features_rank = np.load(os.path.join("../../", dir_ranks, file))
                    #features_rank = np.flip(features_rank) if met ==4 else features_rank
                    unimportant_features_instance = features_rank[:, k:] #works for constant k real world datasets
                    if "syn" in dataset:
                        important_features_instance_onehot = np.zeros_like(features_rank)
                        for i in range(len(features_rank)):
                             imp_feat = features_rank[i, :k+1]
                             important_features_instance_onehot[i, imp_feat]=1

                    #pruning local
                    print(f"unimportant features shape: {unimportant_features_instance.shape}")
                    print("pruning local")
                    for i, data in enumerate(test):
                        test[i, unimportant_features_instance[i]] = Xtst_means[unimportant_features_instance[i]]


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
                        checkpoint_model = np.load(os.path.join("checkpoints", file), allow_pickle=True)
                        print("Loaded original checkpoint: ", file)
                model.load_state_dict(checkpoint_model[()][0], strict=False)
                # testing the subset of features on a trained model
                y_pred = model(torch.Tensor(test))
                y_pred = torch.argmax(y_pred, dim=1)
                accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == ytst) / len(ytst))
                print("test accuracy: ", accuracy)

                #######################
                # for synthetic we compute mcc

                if "syn" in dataset:
                    mcc = matthews_corrcoef(important_features_instance_onehot.flatten(), datatypes_tst.flatten())
                    print("MCC: ", mcc)


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