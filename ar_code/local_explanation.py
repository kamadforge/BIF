"""
Test learning instancewise feature importance

Structure:
(a) we consider three networks, where
(b) we train a baseline network using raw input/output pairs by reducing cross-entropy loss
(c) we then train a switch network together with a predictor network
    by reducing the change in loss of the baseline network and the predictor network

Note that before running this code, run Test_Adult_data.py with alternating dataset.
"""


__author__ = 'anon_m'

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
import socket

from pathlib import Path
import sys
import os
import socket
from data.synthetic_data_loader import synthetic_data_loader
from models.switch_MLP import ThreeNet
from models.nn_3hidden import FC_net, FC
from evaluation_metrics import binary_classification_metrics, compute_median_rank



########################################
# Path
cwd = os.getcwd()
# cwd_parent = Path(__file__).parent.parent
pathmain = cwd
path_code = os.path.join(pathmain, "code")

# if socket.gethostname():
#     pathmain=cwd_parent
#     path_code=cwd

########################################
# Arguments

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="nonlinear_additive") #xor, orange_skin, nonlinear_additive, alternating
    parser.add_argument("--mini_batch_size", default=110, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--kl_term", default=False)
    parser.add_argument("--num_Dir_samples", default=0, type=int)
    parser.add_argument("--point_estimate", default=True)
    parser.add_argument("--mode", default="training")
    # parser.add_argument("--set_hooks", default=True)

    args = parser.parse_args()

    return args

def loss_function(prediction, baseline_net_output, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, kl_term, pre_phi):

    loss = nn.CrossEntropyLoss()

    BCE = loss(prediction, true_y)

    # BCE_baseline = loss(baseline_net_output, true_y)
    #
    # Diff_BCE = (BCE-BCE_baseline)
    # print('BCE', BCE)
    # print('BCE_baseline', BCE_baseline)

    if kl_term:
        # KLD term
        alpha_0 = torch.Tensor([alpha_0])
        hidden_dim = torch.Tensor([hidden_dim])

        # mini_batch_size = phi_cand.shape[0]
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

        # print('KLD and BCE', [KLD, BCE])

        return BCE + KLD
        # return BCE + annealing_rate*KLD/how_many_samps
        # return BCE + annealing_rate * KLD

        # # test L1 norm
        # L1norm_phi = torch.sum(torch.abs(phi_cand))/mini_batch_size/hidden_dim
        # # L1norm_phi = torch.sum(torch.abs(pre_phi)) / mini_batch_size / hidden_dim
        # # print('L1norm and BCE', [L1norm_phi, BCE])
        # return BCE + L1norm_phi

    else:

        # return Diff_BCE
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



def main():

    args = get_args()
    dataset = args.dataset
    mini_batch_size = args.mini_batch_size
    point_estimate = args.point_estimate

    ###########################################33
    # LOAD DATA

    x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset)

    #normalize x_tot
    x_tot / np.expand_dims(np.linalg.norm(x_tot, axis=1), 1)


    # unpack data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)

    X = x_tot[:N, :]
    y = y_tot[:N]
    if dataset == "alternating":
        datatypes = datatypes_tot[:N] #only for alternating, if datatype comes from orange_skin or nonlinear
    else:
        datatypes = None

    X_test = x_tot[N:, :]
    y_test = y_tot[N:]
    if dataset == "alternating":
        datatypes_test = datatypes_tot[N:]

    input_dim = d
    hidden_dim = input_dim
    how_many_samps = N
    output_dim = 2

    #######################################################
    # preparing variational inference to learn switch vector

    alpha_0 = args.alpha #0.01 # below 1 so that we encourage sparsity. #dirichlet dist parameters
    num_repeat = 1 # repeating the entire experiment

    # noise
    # iter_sigmas = np.array([0., 1., 10., 50., 100.])
    iter_sigmas = np.array([0.])
    num_samps_for_switch = args.num_Dir_samples

    # load the baseline network
    method = 'nn'
    baseline_net_trained = np.load(
        os.path.join(path_code, 'models/%s_%s_LR_model' % (dataset, method) + str(int(iter_sigmas[0])) + '.npy'),
        allow_pickle=True)

    which_norm = 'weight_norm'
    baseline_net = FC_net(input_dim, output_dim, 400, which_norm) # hidden_dim = 200
    baseline_net.load_state_dict(baseline_net_trained[()][0], strict=False)

    classifier_net = FC_net(input_dim, output_dim, 400, which_norm)
    classifier_net.load_state_dict(baseline_net_trained[()][0], strict=False)

    switch_net = FC_net(input_dim, input_dim, 200, which_norm) # third input is hidden_dim

    # other_parameters = list(classifier_net.parameters()) + list(switch_net.parameters())
    other_parameters = list(switch_net.parameters())

    # other_parameters = other_parameters + list(baseline_net.parameters())

    for k in range(iter_sigmas.shape[0]):

        # posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
        # switch_parameter_mat = np.empty([num_repeat, input_dim])

        for repeat_idx in range(num_repeat):

            print(repeat_idx)

            model = ThreeNet(baseline_net, classifier_net, switch_net, input_dim, output_dim, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)
            # baseline_net, input_num, output_num, num_samps_for_switch, mini_batch_size, point_estimate)

            print('Starting Training')

            optimizer = optim.SGD(params=other_parameters, lr=0.01, momentum=0.9)
            # optimizer = optim.Adam(params=other_parameters, lr=0.01)
            # optimizer = optim.Adam(model.parameters(), lr=1e-1)


            how_many_epochs = args.epochs
            how_many_iter = np.int(how_many_samps/mini_batch_size)
            training_loss_per_epoch = np.zeros(how_many_epochs)
            # annealing_steps = float(8000.*how_many_epochs)
            annealing_steps = float(how_many_epochs)
            beta_func = lambda s: min(s, annealing_steps) / annealing_steps

            # for name,par in model.named_parameters():
            #     print (name)

            yTrain, xTrain, datatypesTrain = shuffle_data(y, X, how_many_samps, datatypes)


            for epoch in range(how_many_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                annealing_rate = beta_func(epoch)

                for i in range(how_many_iter):

                    # get the inputs
                    inputs = xTrain[i*mini_batch_size:(i+1)*mini_batch_size,:]
                    labels = yTrain[i*mini_batch_size:(i+1)*mini_batch_size]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs, phi_cand, S_cand, pre_phi, baseline_net_output = model(torch.Tensor(inputs), mini_batch_size) #100,10,150

                    labels = torch.squeeze(torch.LongTensor(labels))
                    loss = loss_function(outputs, baseline_net_output, labels, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, args.kl_term, pre_phi=pre_phi)

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                    if i % how_many_iter ==0:
                        print("phi values: ", phi_cand[0,:])
                        print("S values: ", S_cand[0, :])

                # training_loss_per_epoch[epoch] = running_loss/how_many_samps

                training_loss_per_epoch[epoch] = running_loss/how_many_iter
                print('epoch number is ', epoch)
                print('running loss is ', running_loss)

            print('Finished global Training')

            #################################################
            #
            # switch_parameter_mat[repeat_idx,:] = phi_cand.detach().numpy()
            # posterior_mean_switch_mat[repeat_idx,:] = S_cand.detach().numpy()


            #torch.save(model.state_dict(),f"models/switches_{args.dataset}_switch_nn_{args.switch_nn}_local_{args.training_local}.pt")

            ########################
            # test

            if args.mode == "test":

                #############################
                # running the test

                def test_instance(dataset, switch_nn, training_local):

                    print(f"dataset: {dataset}")

                    # path = f"models/switches_{dataset}_switch_nn_{switch_nn}_local_{training_local}.pt"

                    i = 0  # choose a sample
                    mini_batch_size = 2000
                    datatypes_test_samp = None

                    # if switch_nn == False:
                    #     model = Modelnn(d, 2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)
                    # else:
                    #     model = Model_switchlearning(d, 2, num_samps_for_switch, mini_batch_size,
                    #                                  point_estimate=point_estimate)
                    #
                    # model.load_state_dict(torch.load(path), strict=False)

                    inputs_test_samp = X_test[i * mini_batch_size:(i + 1) * mini_batch_size,
                                       :]  # (mini_batch_size* feat_num)
                    labels_test_samp = y_test[i * mini_batch_size:(i + 1) * mini_batch_size]
                    if dataset == "alternating" or "syn" in dataset:
                        datatypes_test_samp = datatypes_test[i * mini_batch_size:(i + 1) * mini_batch_size]

                    if "syn" in dataset:
                        relevant_features = []
                        for i in range(datatypes_test_samp.shape[0]):
                            relevant_features.append(np.where(datatypes_test_samp[i] > 0))
                        datatypes_test_samp = np.array(relevant_features).squeeze(1)

                    inputs_test_samp = torch.Tensor(inputs_test_samp)

                    model.eval()

                    #outputs, phi, S, phi_est = model(inputs_test_samp, mini_batch_size)

                    pred_label, phi_estimate, S_estimate, pre_phi_est, baseline_net_pred = model.forward(inputs_test_samp, mini_batch_size)

                    torch.set_printoptions(profile="full")

                    samples_to_see = 2
                    if mini_batch_size > samples_to_see and datatypes_test_samp is not None:
                        print(datatypes_test_samp[:samples_to_see])
                        print("outputs", outputs[:samples_to_see])
                        print("phi", phi_estimate[:samples_to_see])
                    return S_estimate, datatypes_test_samp



                # inputs_test_samp = torch.Tensor(inputs_test_samp)
                #
                # pred_label, phi_estimate, S_estimate, pre_phi_est, baseline_net_pred = model.forward(inputs_test_samp, mini_batch_size)
                # print('true test labels:', labels_test_samp)
                # print('pred labels: ', torch.argmax(pred_label, dim=1).detach().numpy())
                # print('baseline pred labels: ', torch.argmax(baseline_net_pred, dim=1).detach().numpy())
                # print('estimated switches are:', S_estimate)

                S, datatypes_test_samp = test_instance(dataset, True, False)
                if dataset == "xor":
                    k = 2
                elif dataset == "orange_skin" or dataset == "nonlinear_additive":
                    k = 4
                elif dataset == "alternating":
                    k = 5
                elif dataset == "syn4":
                    k = 7
                elif dataset == "syn5" or dataset == "syn6":
                    k = 9

                median_ranks = compute_median_rank(S, k, dataset, datatypes_test_samp)
                mean_median_ranks = np.mean(median_ranks)
                tpr, fdr = binary_classification_metrics(S, k, dataset, mini_batch_size, datatypes_test_samp)
                print("mean median rank", mean_median_ranks)
                print(f"tpr: {tpr}, fdr: {fdr}")




            # model.eval()
            # print(datatypes_test_samp)
            # outputs, phi, S = model(inputs_test_samp, mini_batch_size)
            # print("outputs", outputs)
            # print("phi", phi)


        # print(filename, filename_phi)
        # np.save(filename,posterior_mean_switch_mat)
        # np.save(filename_last,posterior_mean_switch_mat)
        # np.save(filename_phi, switch_parameter_mat)

if __name__ == '__main__':
    main()
