"""
Test learning instancewise feature importance

Structure:
(a) we consider three networks, where
(b) we train a baseline network using raw input/output pairs by reducing cross-entropy loss
(c) we then train a switch network together with a predictor network
    by reducing the change in loss of the baseline network and the predictor network

Note that before running this code, run Test_Adult_data.py with alternating dataset.
"""


__author__ = 'mijung'

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
from data.synthetic_data_loader import synthetic_data_loader
from models.switch_MLP import ThreeNet


########################################
# Path
cwd = os.getcwd()
cwd_parent = Path(__file__).parent.parent
pathmain = cwd
path_code = os.path.join(pathmain, "code")

########################################
# Arguments

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="alternating") #xor, orange_skin, nonlinear_additive, alternating
    parser.add_argument("--mini_batch_size", default=200, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--alpha", default=0.001, type=float)
    parser.add_argument("--kl_term", default=False)
    parser.add_argument("--num_Dir_samples", default=0, type=int)
    parser.add_argument("--point_estimate", default=True)

    args = parser.parse_args()

    return args

def loss_function(prediction, baseline_net_output, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, kl_term, pre_phi):

    loss = nn.CrossEntropyLoss()

    BCE = loss(prediction, true_y)

    BCE_baseline = loss(baseline_net_output, true_y)

    Diff_BCE = abs(BCE-BCE_baseline)

    if kl_term:
        # KLD term
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

        # trm1_mul = torch.lgamma(torch.sum(phi_cand, dim=1)) - torch.lgamma(hidden_dim * alpha_0)
        # trm2_mul = - torch.sum(torch.lgamma(phi_cand), dim=1) + hidden_dim * torch.lgamma(alpha_0)
        # trm3_mul = torch.sum((phi_cand - alpha_0) * (torch.digamma(phi_cand) - torch.digamma(torch.sum(phi_cand,dim=1)).unsqueeze(dim=1)), dim=1)
        #
        # KL_mul = trm1_mul + trm2_mul + trm3_mul
        # KLD = torch.mean(KL_mul)

        # print('KLD and BCE', [KLD, BCE])

        # return BCE + KLD
        # return BCE + annealing_rate*KLD/how_many_samps
        # return BCE + annealing_rate * KLD

        # test L1 norm
        L1norm_phi = torch.sum(torch.abs(phi_cand))/mini_batch_size/hidden_dim
        # L1norm_phi = torch.sum(torch.abs(pre_phi)) / mini_batch_size / hidden_dim
        # print('L1norm and BCE', [L1norm_phi, BCE])
        return BCE + 10*L1norm_phi

    else:

        return Diff_BCE

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
    baseline_net = np.load(
        os.path.join(path_code, 'models/%s_%s_LR_model' % (dataset, method) + str(int(iter_sigmas[0])) + '.npy'),
        allow_pickle=True)

    for k in range(iter_sigmas.shape[0]):

        # posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
        # switch_parameter_mat = np.empty([num_repeat, input_dim])

        for repeat_idx in range(num_repeat):

            print(repeat_idx)

            model = ThreeNet(baseline_net, 2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)

            print('Starting Training')

            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

                    # if i % how_many_iter ==0:
                    #     print("switch: ", S_cand[0,:])

                # training_loss_per_epoch[epoch] = running_loss/how_many_samps

                training_loss_per_epoch[epoch] = running_loss
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

            i=0 #samples number
            mini_batch_size = 5
            inputs_test_samp = X_test[i * mini_batch_size:(i + 1) * mini_batch_size, :]
            labels_test_samp = y_test[i * mini_batch_size:(i + 1) * mini_batch_size]
            datatypes_test_samp = datatypes_test[i * mini_batch_size:(i + 1) * mini_batch_size]


            inputs_test_samp = torch.Tensor(inputs_test_samp)

            pred_label, phi_estimate, S_estimate, pre_phi_est = model.forward(inputs_test_samp, mini_batch_size)
            print('true test labels:', labels_test_samp)
            print('pred labels: ', torch.argmax(pred_label, dim=1).detach().numpy())
            print('estimated switches are:', S_estimate)

            #
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
