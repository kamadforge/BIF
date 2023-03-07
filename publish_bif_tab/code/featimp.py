# """
# 1. removed vips data
# """
#
# __author__ = 'mijung'
#
# import os
# import sys
# import scipy
# import scipy.io
# import numpy as np
# import numpy.random as rn
# from sklearn.metrics import roc_curve,auc
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import pickle
# import autodp
from autodp import privacy_calibrator
#from models.nn_3hidden import FC, FC_net
# from itertools import combinations, chain
# import argparse
#
# import torch.nn as nn
# import torch.optim as optim
# import torch

#mvnrnd = rn.multivariate_normal
#
# import sys
# from data.tab_dataloader import load_adult, load_credit, load_adult_short
# from data.make_synthetic_datasets import generate_data
# from data.make_synthetic_datasets import generate_invase
# #from data.synthetic_data_loader import synthetic_data_loader
#
# For instancewise training one can do two things:

# 1. train the switch vector by backpropagation (the original way) and then finetune on one example (local training)
#  set training_local to True
#  set switch_nn to False

# 2. Train the switch network which outputs the switch parameters (phi) and feeds them on the classifier
# set switch_nn to True
# possibly set_hooks to False
# set training_local to False


# when adding a new dataset make sure the number of test samples is the same for min(y_test, 100000) takes y_pred


__author__ = 'anom_m'

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
from pathlib import Path
import sys
import os
import socket
import sys
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).resolve().parent / "data"))
from tab_dataloader import load_adult, load_credit, load_adult_short, load_intrusion
from synthetic_data_loader import synthetic_data_loader
from evaluation_metrics import compute_median_rank, binary_classification_metrics
from models.nn_3hidden import FC

from train_network import get_data


########################################
# PATH

cwd = os.getcwd()
cwd_parent = Path(__file__).parent.parent
if socket.gethostname()=='worona.local':
    pathmain = cwd
    path_code = os.path.join(pathmain, "code")
elif 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    sys.path.append(os.path.join(cwd_parent, "data"))
    pathmain=cwd
    path_code = os.path.join(pathmain, "code")
    #path_code = os.path.join(pathmain)
else:
    pathmain = cwd_parent
    path_code = cwd

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints_bif", exist_ok=True)

##################################################3
# ARGUMENTS

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--dataset", default="intrusion") #xor, orange_skin, nonlinear_additive, alternating, syn4, syn5, syn6, adult_short, credit, intrusion
    parser.add_argument("--load_dataset", default=1, type=int)
    parser.add_argument("--method", default="nn")
    parser.add_argument("--mini_batch_size", default=200, type=int)
    parser.add_argument("--epochs", default=7, type=int) # 7
    parser.add_argument("--lr", default=0.1, type=float)
    # for switch training
    parser.add_argument("--num_Dir_samples", default=30, type=int)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--point_estimate", default=1, type=int)
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--test", default=1, type=int)
    # for instance wise training switch_nn=1, and 0 for global
    parser.add_argument("--switch_nn", default=1, type=int)
    parser.add_argument("--training_local", default=0, type=int)
    parser.add_argument("--local_training_iter", default=200, type=int)
    parser.add_argument("--set_hooks", default=1, type=int)
    parser.add_argument("--kl_term", default=0, type=int)

    parser.add_argument("--ktop_real", default=3, type=int)
    parser.add_argument("--runs_num", default=2, type=int)
    # parse
    args = parser.parse_args()
    return args





#######################
# model local or global

from models.switch_MLP import Model_switchlearning
from models.switch_MLP import Modelnn
# settings
torch.set_printoptions(precision=4, sci_mode=False)

#######################
# LOSS

def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method, kl_term, point_estimate, args):
    if 1:
        if args.switch_nn: #local explanations
            loss = nn.CrossEntropyLoss()
            if point_estimate:
                # mini_batch x [0,1] vs. mini_batch x [0,1] for BCE
                if args.mini_batch_size == 1:
                    true_y = true_y.unsqueeze(0)
                BCE = loss(prediction, true_y)
            else: #sampling
                #for each sample compute the loss with the same true_y, then take the mean of the losses
                # computing the crossnetropy term of the elbo
                BCE_mat = torch.zeros(prediction.shape[1])  # contains losses for each sample
                for ind in torch.arange(0, prediction.shape[1]):  # for each sample
                    # mini_batch x samples x [0,1] and for each sample mini_batch x [0,1]
                    y_pred = prediction[:, ind, :]  # get the prediction
                    BCE_mat[ind] = loss(y_pred, true_y)  # compute the loss for that sample
                BCE = torch.mean(BCE_mat)  # average the losses
            if kl_term:
                if point_estimate:
                    # KLD termaa
                    alpha_0 = torch.Tensor([alpha_0])
                    hidden_dim = torch.Tensor([hidden_dim])
                    trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
                    trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
                    trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))
                    KLD = trm1 + trm2 + trm3
                    return BCE + annealing_rate*KLD/how_many_samps
                else: # sampling with kl
                    alpha_0 = torch.Tensor([alpha_0])
                    hidden_dim = torch.Tensor([hidden_dim])
                    mini_batch_size = phi_cand.shape[0]
                    # KL_mat = torch.zeros(mini_batch_size)
                    # for i in torch.arange(0,mini_batch_size):
                    #     phi = phi_cand[i,:]
                    #     trm1 = torch.lgamma(torch.sum(phi)) - torch.lgamma(hidden_dim*alpha_0)
                    #     trm2 = - torch.sum(torch.lgamma(phi)) + hidden_dim*torch.lgamma(alpha_0)
                    #     trm3 = torch.sum((phi-alpha_0)*(torch.digamma(phi)-torch.digamma(torch.sum(phi))))
                    #
                    #     KL_mat[i] = trm1 + trm2 + trm3
                    #
                    trm1_mul = torch.lgamma(torch.sum(phi_cand, dim=1)) - torch.lgamma(hidden_dim * alpha_0)
                    trm2_mul = - torch.sum(torch.lgamma(phi_cand), dim=1) + hidden_dim * torch.lgamma(alpha_0)
                    trm3_mul = torch.sum((phi_cand - alpha_0) * (torch.digamma(phi_cand) - torch.digamma(torch.sum(phi_cand,dim=1)).unsqueeze(dim=1)), dim=1)
                    KL_mul = trm1_mul + trm2_mul + trm3_mul
                    KLD = torch.mean(KL_mul)
                    # return BCE + + annealing_rate * KLD / how_many_samps
                    return BCE + KLD / mini_batch_size
            else: # no kl term (both point estimate and sampling)
                return BCE
        else: #global explanations (non- importance switch nn)
            loss = nn.CrossEntropyLoss()
            if point_estimate:
                if args.mini_batch_size == 1:
                    true_y = true_y.unsqueeze(0)
                BCE = loss(prediction, true_y)
            else:  # sampling
                BCE_mat = torch.zeros(prediction.shape[1])
                for ind in torch.arange(0, prediction.shape[1]):
                    y_pred = prediction[:, ind, :]
                    BCE_mat[ind] = loss(y_pred, true_y)
                BCE = torch.mean(BCE_mat)
            if kl_term:
                # KLD term
                alpha_0 = torch.Tensor([alpha_0])
                hidden_dim = torch.Tensor([hidden_dim])
                trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim * alpha_0)
                trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim * torch.lgamma(alpha_0)
                trm3 = torch.sum(
                    (phi_cand - alpha_0) * (torch.digamma(phi_cand) - torch.digamma(torch.sum(phi_cand))))
                KLD = trm1 + trm2 + trm3
                return BCE + + annealing_rate * KLD / how_many_samps
            else: #no kl term (both point estimate and sampling)
                return BCE

def shuffle_data(y,x,how_many_samps, datatypes=None):
    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]
    if datatypes is None:
        shuffled_datatypes = None
    else:
        shuffled_datatypes = datatypes[idx]
    return shuffled_y, shuffled_x, shuffled_datatypes


def train_switches(args, loaded_model, X, Xtst, y, ytst, datatypes_tr, datatypes_tst):

    if args.dataset == "intrusion":
        output_num = 4
    else:
        output_num = 2

    num_samples_tr, num_feat = X.shape
    mean_of_means = np.zeros(num_feat)
    num_repeat = 1 # repeating the entire experiment

    for repeat_idx in range(num_repeat):
        print(repeat_idx)
        ####################
        # get combined model (g+switches, based on the choice of local or global feature selection) and load the model g, and train switches
        if args.switch_nn == False:
            model = Modelnn(num_feat, output_num, args.num_Dir_samples, args.mini_batch_size, point_estimate=args.point_estimate)
        else:
            model = Model_switchlearning(num_feat, output_num, args.num_Dir_samples, args.mini_batch_size,point_estimate=args.point_estimate)
        # load model g and freeze it
        model.load_state_dict(loaded_model[()][repeat_idx], strict=False)
        if args.set_hooks:
            # in case you use pre-trained classifier
            h = model.fc1.weight.register_hook(lambda grad: grad * 0)
            h = model.fc2.weight.register_hook(lambda grad: grad * 0)
            h = model.fc4.weight.register_hook(lambda grad: grad * 0)
            # h = model.bn1.weight.register_hook(lambda grad: grad * 0)
            # h = model.bn2.weight.register_hook(lambda grad: grad * 0)
            h = model.fc1.bias.register_hook(lambda grad: grad * 0)
            h = model.fc2.bias.register_hook(lambda grad: grad * 0)
            h = model.fc4.bias.register_hook(lambda grad: grad * 0)
            # h = model.bn1.bias.register_hook(lambda grad: grad * 0)
            # h = model.bn2.bias.register_hook(lambda grad: grad * 0)

        ############################################
        # perform training
        print('Starting Switch Training')
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
        how_many_epochs = args.epochs
        how_many_iter = np.int(num_samples_tr / args.mini_batch_size)
        training_loss_per_epoch = np.zeros(how_many_epochs)
        annealing_steps = float(8000. * how_many_epochs)
        beta_func = lambda s: min(s, annealing_steps) / annealing_steps
        yTrain, xTrain, datatypesTrain = shuffle_data(y, X, num_samples_tr, datatypes_tr)
        # loop over the dataset multiple times
        for epoch in range(how_many_epochs):
            running_loss = 0.0
            annealing_rate = beta_func(epoch)
            for i in range(how_many_iter):
                # get the inputs
                inputs = xTrain[i * args.mini_batch_size:(i + 1) * args.mini_batch_size, :]
                labels = yTrain[i * args.mini_batch_size:(i + 1) * args.mini_batch_size]
                if (args.dataset == "alternating" or "syn" in args.dataset):
                    datatypes_train_batch = datatypesTrain[i * args.mini_batch_size:(i + 1) * args.mini_batch_size]
                optimizer.zero_grad()
                # run the model
                outputs, phi_cand, S, prephi, var_phi = model(torch.Tensor(inputs),
                                                              args.mini_batch_size)  # the example shape 100,10,150
                # loss
                labels = torch.squeeze(torch.LongTensor(labels))
                loss = loss_function(outputs, labels, phi_cand, args.alpha, num_feat, num_samples_tr, annealing_rate,args.method, args.kl_term, args.point_estimate, args)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # print
                if i % how_many_iter == 0:
                    if args.switch_nn:  # instancewise, averaging over all the exmaples
                        print("Local setting:")  # local
                        if args.point_estimate:
                            print("Mean over importance vectors batch: ", S.mean(dim=0))
                        else:
                            print("Mean over batch and samples", S.mean(dim=0).mean(dim=1))  # batch x feat x samtc
                    else:  # global (no switch nn)
                        print(("Global setting:"))
                        if args.point_estimate:
                            print("One importance vector", S)
                            print(torch.argsort(S)[::-1])
                            S_global_final = S
                        else:
                            print("Mean over samples", S.mean(dim=0))
                            print(torch.argsort(S.mean(dim=0), descending=True))
                            S_global_final = S.mean(dim=0)
                        print(f"Variance: {var_phi}")

            training_loss_per_epoch[epoch] = running_loss
            print('epoch number is ', epoch)
            print('running loss is \n', running_loss)
            # if global save the rank
        if not args.switch_nn:
            torch.save(S_global_final,
                       f"rankings/global/global_{args.dataset}_pointest_{args.point_estimate}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt")
            print("Global switch saved")

        # print('Finished global Training\n')

        if not args.switch_nn:
            estimated_params = list(model.parameters())
            """ posterior mean over the switches """
            # num_samps_for_switch
            phi_est = F.softplus(torch.Tensor(estimated_params[0]))
            print('estimated parameters are ', phi_est.detach().numpy())
            print("-" * 20)
            # switch_parameter_mat[repeat_idx,:] = phi_est.detach().numpy()
            concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
            # beta_param = torch.ones(self.hidden_dim,1).repeat(1,num_samps)
            beta_param = torch.ones(concentration_param.size())
            Gamma_obj = Gamma(concentration_param, beta_param)
            gamma_samps = Gamma_obj.rsample()
            Sstack = gamma_samps / torch.sum(gamma_samps, 0)
            avg_S = torch.mean(Sstack, 1)
            std_S = torch.std(Sstack, 1)
            posterior_mean_switch = avg_S.detach().numpy()
            posterior_std_switch = std_S.detach().numpy()
            # posterior_mean_switch_mat[repeat_idx,:] = posterior_mean_switch
            print('estimated posterior mean of Switch is', posterior_mean_switch)
            mean_of_means += posterior_mean_switch

            torch.save(model.state_dict(),
                       os.path.join(path_code,
                                    f"checkpoints_bif/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt"))

        else:  # if switch_nn is true testing a single instance

            torch.save(model.state_dict(),
                       os.path.join(path_code,
                                    f"checkpoints_bif/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt"))

####
# testing

#####################
# FIRST MAKE A RUN TO GET LOCAL IMPORTANCE (FOR LOCAL)
# that is both for synthetic and real-world datasets

def test_get_switches(dataset, switch_nn, training_local, output_num, X_test, y_test, datatypes_test, args):

    # get data sample, x, y, and gt_features
    print(f"dataset: {dataset}")
    _, num_feat = X_test.shape
    path = os.path.join(path_code, f"checkpoints_bif/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt")
    i = 0  # choose a sample
    mini_batch_size = X_test.shape[0]  # entire test dataset
    inputs_test_samp = X_test[i * mini_batch_size:(i + 1) * mini_batch_size, :]  # (mini_batch_size* feat_num)
    labels_test_samp = y_test[i * mini_batch_size:(i + 1) * mini_batch_size]
    if dataset == "alternating" or "syn" in dataset:
        datatypes_test_samp = datatypes_test[i * mini_batch_size:(i + 1) * mini_batch_size]

    # choose global or local switch+model g
    if switch_nn:
        model = Model_switchlearning(num_feat,output_num, args.num_Dir_samples, mini_batch_size, point_estimate=args.point_estimate)
    else:
        model = Modelnn(num_feat,output_num, args.num_Dir_samples, mini_batch_size, point_estimate=args.point_estimate)
    model.load_state_dict(torch.load(path), strict=False)
    print(f"\nModel with importance network loaded from {path}")

    # ground truth relevant features (used for synthtic data only)
    # relevant_features - indices
    # datatypes_test_samp - one-hot vector
    if "syn" in dataset:
        relevant_features=[]
        for i in range(datatypes_test_samp.shape[0]):
            relevant_features.append(np.where(datatypes_test_samp[i]>0))
        datatypes_test_samp_arg = np.array(relevant_features).squeeze(1)
    else:
        datatypes_test_samp_arg = None
        datatypes_test_samp = None

    # run the forward test on the original all features to get the S importance values
    inputs_test_samp = torch.Tensor(inputs_test_samp)
    model.eval()
    outputs, phi, S, phi_est, var_phi = model(inputs_test_samp, mini_batch_size)
    torch.set_printoptions(profile="full")

    return S, datatypes_test_samp_arg, datatypes_test_samp, inputs_test_samp

#########################3
# PRUNE AND TEST MODEL G ON SUBSET OF FEATURES

##################
# REAL DATASETS

def test_pruned(S, inputs_test_samp, ktop, model_g, X_test, y_test, args):

    global X_test_means
    X_test_means = np.mean(X_test, 0)
    print(X_test_means)

    if args.dataset == "intrusion":
        output_num = 4
    else:
        output_num = 2

    inputs_test_samp1=inputs_test_samp.clone()
    os.makedirs("rankings", exist_ok=True)
    os.makedirs("rankings/local", exist_ok=True)
    os.makedirs("rankings/global", exist_ok=True)

    if args.switch_nn: #local
        instance_best_features_ascending = np.argsort(S.detach().cpu().numpy(), axis=1)
        instance_unimportant_features = instance_best_features_ascending[:, :-ktop] #(num_test_samples, num_features, num_dirichlet_samples)
    else: #global
        instance_best_features_ascending = np.argsort(S.detach().cpu().numpy())
        instance_best_features_ascending = instance_best_features_ascending[:-ktop]
        instance_unimportant_features = np.tile(instance_best_features_ascending, (inputs_test_samp1.shape[0], 1))
    print("unimportant features shape", instance_unimportant_features.shape)


    # run the test (forward pass) on the subset of features, which were selected, the rest is pruned/zeroed out/ mean
    for i, data in enumerate(inputs_test_samp1):
        inputs_test_samp1[i, instance_unimportant_features[i]] = torch.Tensor(X_test_means[instance_unimportant_features[i]])
        #inputs_test_samp1[i, instance_unimportant_features[i]] = 0

    _ ,feature_num = X_test.shape
    input_num= feature_num
    model = FC(input_num, output_num)
    LR_model = np.load(model_g, allow_pickle=True)
    model.load_state_dict(LR_model[()][0], strict=False)
    # run the original model g on the pruned features
    y_pred = model(torch.Tensor(inputs_test_samp1))
    y_pred = torch.argmax(y_pred, dim=1)

    accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == y_test) / len(y_test))
    print(f"model g test accuracy top-{ktop} features: ", accuracy)

    return accuracy

def test_pruned_syn(S, args, X_test, ytst, datatypes_tst, datatypes_test_samp_arg, synthetic):


    k_dic = {"xor": 2, "subtract": 2, "xor_mean5": 2, "orange_skin": 4, "orange_skin_mean5": 4, "nonlinear_additive": 4, "alternating": 5, "syn4": 7, "syn4_mean5": 7, "syn5": 9, "syn6": 9}
    k = k_dic[args.dataset]

    if (args.switch_nn):
        # if not args.point_estimate:
        #     S = S.mean(dim=2)

        median_ranks = compute_median_rank(S, k, args.dataset, datatypes_test_samp_arg)
        mean_median_ranks = np.mean(median_ranks)
        # if not args.point_estimate:
        #    S=S.mean(dim=1)
        mini_batch_size = 2000

        tpr, fdr, mcc = binary_classification_metrics(S, k, args.dataset, mini_batch_size,
                                                      datatypes_test_samp_arg, args.switch_nn)
        print("mean median rank", mean_median_ranks)
        # print(f"tpr: {tpr}, fdr: {fdr}")
        print(f"mcc: {mcc}")
    else:
        if (args.dataset in synthetic):
            mini_batch_size = 2000
            #if not args.point_estimate:
            #    S = S.mean(dim=0)

            S = np.tile(S.detach().cpu().numpy(), (2000, 1))
            print(S[0:5])

            tpr, fdr, mcc = binary_classification_metrics(S, k, args.dataset, mini_batch_size, datatypes_test_samp_arg, True)

            # print("mean median rank", mean_median_ranks)
            print(f"tpr: {tpr}, fdr: {fdr}")
            print(f"mcc: {mcc}")
        else:  # real datasets, no tpr, fdr, mcc can be calculated
            tpr, fdr = -1, -1

    return mcc


#######################################################
# MAIN

def main():



    ###########################################33
    # LOAD DATA

    synthetic = ["xor", "xor_mean5", "orange_skin", "orange_skin_mean5", "nonlinear_additive", "alternating", "syn4", "syn4_mean5", "syn5", "syn6"]

    X, X_test, y, ytst, x_tot, y_tot, datatypes_tr, datatypes_tst, datatypes_tst_num_relevantfeatures = get_data(args, synthetic)

    # global X_test_means
    # X_test_means = np.mean(X_test, 0)
    # print(X_test_means)



    #######################################################
    # VARIATIONAL INFERENCE PARAMS TO LEARN IMPORTANCE SWITCH




    #####################################
    # load pretrained model g for the switch model
    for i in os.listdir("checkpoints"):
        file = os.path.join(path_code, "checkpoints", i)
        if args.dataset in file and args.method in file:
            loaded_model = np.load(file, allow_pickle=True)
            model_g = file # assigning here this args.dataset checkpoint bc for loop is going through all the checkpoints incl other datasets
            print("Loaded: ", file)
    if not os.path.isdir("weights"):
        os.mkdir("weights")

    # train mode
    if args.train:
        train_switches(args, loaded_model, X, X_test, y, ytst, datatypes_tr, datatypes_tst)

####################################3
# TEST



    if args.test:

            print("\nTesting:\n")


            if args.switch_nn:
                # getting lcaol switches from the importance net
                S, datatypes_test_samp_arg, datatypes_test_samp_onehot, inputs_test_samp = test_get_switches(args.dataset, args.switch_nn, False, output_num, X_test, ytst, datatypes_tst, args)
                print("Got local switches from the importance network")
                if not args.point_estimate:
                    S = torch.mean(S, axis=2)
            else:  # get global switches
                S = torch.load(f"rankings/global/global_{args.dataset}_pointest_{args.point_estimate}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt")
                print(f"Switch global loaded: {S}")
                #S = S.unsqueeze(0).repeat(X_test.shape[0],1)
                inputs_test_samp = torch.Tensor(X_test)
                datatypes_test_samp_arg = None


                # testing the local switches
            synthetic = ["xor", "xor_mean5", "orange_skin", "orange_skin_mean5", "nonlinear_additive", "alternating", "syn4", "syn4_mean5", "syn5", "syn6"]
            if (args.dataset in synthetic):
                accuracy = test_pruned_syn(S, args, X_test, ytst, datatypes_tst, datatypes_test_samp_arg)
            else:
                accuracy = test_pruned(S, inputs_test_samp, args.ktop_real, model_g, X_test, ytst, args)

            print("Tested on the subset of features chosen for each instance")

            return [accuracy]



    else:
        print("Please test the global setting in train_network.py")

            #######################################
            # EVALUATION FOR SYNTHETIC DATASETS
            # we look here only on the local feature choice




    # print('estimated posterior mean of Switch is', estimated_Switch)

    # f = plt.figure(2)
    # plt.plot(np.arange(0, hidden_dim), trueSwitch, 'ko')
    # plt.errorbar(np.arange(0, hidden_dim), posterior_mean_switch, yerr=posterior_std_switch, fmt='ro')
    # # plt.plot(estimated_Switch, 'ro')
    # # plt.plot(posterior_mean_switch, 'ro')
    # plt.title('true Switch (black) vs estimated Switch (red)')
    # plt.show()

    # fig_title =
    # f.savefig("plots/posterior_mean_switch_without_sampling_hidden_dim_500_epoch_400.pdf")
    # f.savefig("plots/posterior_mean_switch_with_sampling_hidden_dim_500_epoch_400.pdf")
    # f.savefig("plots/posterior_mean_switch_with_sampling_hidden_dim_20_epoch_400.pdf")


if __name__ == '__main__':

    args = get_args()
    print(args)

    if "syn" in args.dataset and not args.switch_nn:
        print("\n\n Please use local setting to test alternating datasets")
        exit()

    global output_num
    if args.dataset == "intrusion":
        output_num = 4
    else:
        output_num = 2

    runs = args.runs_num

    tprs, fdrs, Ss = [], [], []
    for i in range(runs):
        print(f"\n\nRun: {i}\n")
        vals = main()

        if vals is None:
            print("no local values returned")
            continue

        if len(vals)==3: # synthetic
            tpr = vals[0]; fdr = vals[1]; S = vals[2]
            tprs.append(tpr); fdrs.append(fdr); Ss.append(S.mean(dim=0).detach().numpy())
        elif len(vals) == 1:  # real
            acc = vals[0]
            tprs.append(acc)


    print("*" * 20)
    if len(vals) == 3:
        S_average = np.round(np.mean(Ss, axis=0), 3)
        S_average_nums = np.argsort(S_average)[::-1]
        print(f"tpr mean {np.mean(tprs)}, fdr mean: {np.mean(fdrs)}, tpr std {np.std(tprs)}, fdr_std {np.std(fdrs)}, S_testmean: {S_average}, S_args: {S_average_nums}")
        print(",".join([str(a) for a in S_average_nums]))

    if len(vals)==1: #real
        print(f"final acc mean {np.mean(tprs)}")
        print(f"final acc std {np.std(tprs)}")
        print(f"dataset {args.dataset}, ktop_real {args.ktop_real} lr {args.lr}, it {args.epochs}, mini_batch_size {args.mini_batch_size}")

    print("********END\n\n\n")



