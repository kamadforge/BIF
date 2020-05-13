"""
Test learning feature importance under DP and non-DP models
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

cwd = os.getcwd()
cwd_parent = Path(__file__).parent.parent
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    sys.path.append(os.path.join(cwd_parent, "data"))
    from data.tab_dataloader import load_cervical, load_adult, load_credit
    pathmain=cwd
    path_code = os.path.join(pathmain, "code")
elif socket.gethostname()=='worona.local':
    pathmain = cwd
    path_code = os.path.join(pathmain, "code")
else:
    from data.tab_dataloader import load_cervical, load_adult, load_credit
    from models.nn_3hidden import FC
    pathmain=cwd_parent
    path_code=cwd


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="xor") #xor, orange_skin, nonlinear_additive
    parser.add_argument("--method", default="nn")
    parser.add_argument("--switch_nn", default=True)
    parser.add_argument("--num_Dir_samples", default=50, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--mini_batch_size", default=110, type=int)
    parser.add_argument("--point_estimate", default=True)
    parser.add_argument("--training_local", default=False)

    args = parser.parse_args()

    return args

args = get_args()


def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method):

    if method=="vips":
        BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

        return BCE
    elif method=="nn":

        loss = nn.CrossEntropyLoss()

        if not args.point_estimate:
            BCE_mat = torch.zeros(prediction.shape[1])
            for ind in torch.arange(0, prediction.shape[1]):
                y_pred = prediction[:,ind,:]
                BCE_mat[ind] = loss(y_pred, true_y)

            BCE = torch.mean(BCE_mat)
        else:
            BCE = loss(prediction, true_y)

        # # KLD term
        # alpha_0 = torch.Tensor([alpha_0])
        # hidden_dim = torch.Tensor([hidden_dim])
        #
        # trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
        # trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
        # trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))
        #
        # KLD = trm1 + trm2 + trm3
        # # annealing kl-divergence term is better
        #
        # #KLD=0 #just to check
        #
        # return BCE + annealing_rate*KLD/how_many_samps
        return BCE


def shuffle_data(y,x,how_many_samps, datatypes=None):

    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]
    if datatypes is None:
        shuffle_datatypes = None
    else:
        shuffled_datatypes = datatypes[idx]

    return shuffled_y, shuffled_x, shuffle_datatypes




def main():


    dataset = args.dataset
    method = args.method
    mini_batch_size = args.mini_batch_size
    point_estimate = args.point_estimate

    if method == "nn":
        from models.switch_MLP import Modelnn
        from models.switch_MLP import Model_switchlearning
    elif method=="vips":
        from models.switch_LR import Model


    ###########################################33
    # load data
    x_tot, y_tot, datatypes = synthetic_data_loader(dataset)

    # unpack data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)

    X = x_tot[:N, :]
    y = y_tot[:N]

    input_dim = d
    hidden_dim = input_dim
    how_many_samps = N

    #######################################################
    # preparing variational inference
    alpha_0 = args.alpha #0.01 # below 1 so that we encourage sparsity.
    num_samps_for_switch = args.num_Dir_samples

    num_repeat = 1
    # iter_sigmas = np.array([0., 1., 10., 50., 100.])
    iter_sigmas = np.array([0.])




    for k in range(iter_sigmas.shape[0]):

        LR_model = np.load(os.path.join(path_code, 'models/%s_%s_LR_model' % (dataset, method) + str(int(iter_sigmas[k])) + '.npy'), allow_pickle=True)

        filename = os.path.join(path_code, 'weights/%s_%d_%.1f_%d_switch_posterior_mean' % (dataset, args.num_Dir_samples, args.alpha, args.epochs)+str(int(iter_sigmas[k])))
        filename_last = os.path.join(path_code, 'weights/%s_switch_posterior_mean' % (dataset)+str(int(iter_sigmas[k])))
        filename_phi = os.path.join(path_code, 'weights/%s_%d_%.1f_%d_switch_parameter' % (dataset, args.num_Dir_samples, args.alpha, args.epochs)+ str(int(iter_sigmas[k])))

        posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
        switch_parameter_mat = np.empty([num_repeat, input_dim])

        mean_of_means=np.zeros(input_dim)
        for repeat_idx in range(num_repeat):
            print(repeat_idx)
            if method=="vips":

                model = Model(input_dim=input_dim, LR_model=torch.Tensor(LR_model[repeat_idx,:]), num_samps_for_switch=num_samps_for_switch)

            elif method=="nn":

                if args.switch_nn==False:
                    model = Modelnn(d,2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)
                else:
                    model = Model_switchlearning(d,2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)
                model.load_state_dict(LR_model[()][repeat_idx], strict=False)

                h = model.fc1.weight.register_hook(lambda grad: grad * 0)
                h = model.fc2.weight.register_hook(lambda grad: grad * 0)
                #h = model.fc3.weight.register_hook(lambda grad: grad * 0)
                h = model.fc4.weight.register_hook(lambda grad: grad * 0)
                h = model.fc1.bias.register_hook(lambda grad: grad * 0)
                h = model.fc2.bias.register_hook(lambda grad: grad * 0)
                #h = model.fc3.bias.register_hook(lambda grad: grad * 0)
                h = model.fc4.bias.register_hook(lambda grad: grad * 0)

            ############################################################################################3

            print('Starting Training')

            #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            optimizer = optim.Adam(model.parameters(), lr=1e-1)
            how_many_epochs = args.epochs
            how_many_iter = np.int(how_many_samps/mini_batch_size)


            training_loss_per_epoch = np.zeros(how_many_epochs)

            annealing_steps = float(8000.*how_many_epochs)
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
                    outputs, phi_cand = model(torch.Tensor(inputs), mini_batch_size) #100,10,150

                    if method=="vips":
                        labels = torch.squeeze(torch.Tensor(labels))
                        loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method)
                    elif method == "nn":
                        labels = torch.squeeze(torch.LongTensor(labels))
                        loss = loss_function(outputs, labels, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method)

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                    #print(model.fc1.weight[1:5])
                    #print(model.fc3.bias[1:5])
                    #print(model.parameter)

                # training_loss_per_epoch[epoch] = running_loss/how_many_samps
                training_loss_per_epoch[epoch] = running_loss
                print('epoch number is ', epoch)
                print('running loss is ', running_loss)

            print('Finished global Training')

            estimated_params = list(model.parameters())
            """ posterior mean over the switches """
            phi_est = F.softplus(torch.Tensor(estimated_params[0]))
            print('estimated parameters are ', phi_est.detach().numpy())

            ###################################################

            if args.training_local:

                print("\nStarting local training")

                how_many_iter = 2000

                annealing_steps_local = float(8000. * how_many_iter)
                beta_func_local = lambda s: min(s, annealing_steps_local) / annealing_steps_local

                #for epoch in range(how_many_epochs):  # loop over the dataset multiple times
                if 1:

                    running_loss = 0.0
                    mini_batch_size =1


                    i=5
                    # get the inputs
                    inputs = xTrain[i * mini_batch_size:(i + 1) * mini_batch_size, :]
                    labels = yTrain[i * mini_batch_size:(i + 1) * mini_batch_size]
                    datatypes = datatypesTrain[i * mini_batch_size:(i + 1) * mini_batch_size]
                    print("Training on:", datatypes)



                    for i in range(how_many_iter):

                        annealing_rate = beta_func_local(how_many_iter)

                        if i % 100 ==0:
                            print(i)


                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs, phi_cand = model(torch.Tensor(inputs), mini_batch_size)  # 100,10,150

                        if method == "vips":
                            labels = torch.squeeze(torch.Tensor(labels))
                            loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand,
                                                 alpha_0, hidden_dim, how_many_samps, annealing_rate, method)
                        elif method == "nn":
                            labels = torch.LongTensor(labels)
                            loss = loss_function(outputs, labels, phi_cand, alpha_0, hidden_dim, how_many_samps,
                                                 annealing_rate, method)
                        loss.backward()
                        optimizer.step()

                        #if i % 100 ==0:
                        #    print(loss)


                        # print statistics
                        running_loss += loss.item()

                        # print(model.fc1.weight[1:5])
                        # print(model.fc3.bias[1:5])
                        # print(model.parameter)

                    # training_loss_per_epoch[epoch] = running_loss/how_many_samps
                    #training_loss_per_epoch[epoch] = running_loss
                    #print('epoch number is ', epoch)
                    print('running loss is ', running_loss)

            #################################################

            if not args.switch_nn:
                estimated_params = list(model.parameters())

                """ posterior mean over the switches """
                # num_samps_for_switch
                phi_est = F.softplus(torch.Tensor(estimated_params[0]))

                print('estimated parameters are ', phi_est.detach().numpy())
                print("-"*20)

                switch_parameter_mat[repeat_idx,:] = phi_est.detach().numpy()

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

                posterior_mean_switch_mat[repeat_idx,:] = posterior_mean_switch
                print('estimated posterior mean of Switch is', posterior_mean_switch)
                mean_of_means+=posterior_mean_switch

            else:



                print('estimated parameters are ', phi_cand.detach().numpy())
                print("-" * 20)

        print(filename, filename_phi)
        print('*'*30)
        print(mean_of_means/num_repeat)
        np.save(filename,posterior_mean_switch_mat)
        np.save(filename_last,posterior_mean_switch_mat)
        np.save(filename_phi, switch_parameter_mat)


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
    main()
