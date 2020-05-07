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

import sys
from data.tab_dataloader import load_cervical, load_adult, load_credit
from models.nn_3hidden import FC

mini_batch_size = 100
method = "nn"
dataset = 'orange_skin'

class Model(nn.Module):

    def __init__(self, input_dim, LR_model, num_samps_for_switch):
    # def __init__(self, input_dim, hidden_dim):
        super(Model, self).__init__()

        self.W = LR_model
        self.parameter = Parameter(-1e-10*torch.ones(input_dim),requires_grad=True)
        self.num_samps_for_switch = num_samps_for_switch

    def forward(self, x): # x is mini_batch_size by input_dim

        phi = F.softplus(self.parameter)

        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)


        """ draw Gamma RVs using phi and 1 """
        num_samps = self.num_samps_for_switch
        concentration_param = phi.view(-1,1).repeat(1,num_samps)
        beta_param = torch.ones(concentration_param.size())
        #Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample() #200, 150, input_dim x samples_num

        if any(torch.sum(gamma_samps,0)==0):
            print("sum of gamma samps are zero!")
        else:
            Sstack = gamma_samps / torch.sum(gamma_samps, 0) # input dim by  # samples

        x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples
        x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W))) #[100,150]
        labelstack = torch.sigmoid(x_out)

        return labelstack, phi



class Modelnn(nn.Module):

    def __init__(self, input_num, output_num, num_samps_for_switch):
    # def __init__(self, input_dim, hidden_dim):
        super(Modelnn, self).__init__()

        #self.W = LR_model
        self.parameter = Parameter(-1e-10*torch.ones(input_num),requires_grad=True)
        self.num_samps_for_switch = num_samps_for_switch

        self.fc1 = nn.Linear(input_num, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, output_num)

    def switch_func_fc(self, output, SstackT):

        #rep = SstackT.unsqueeze(2).unsqueeze(2).repeat(1, 1,)  # (150,10,24,24)
        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output=torch.einsum('ij, mj -> imj', (SstackT, output))
        #output = torch.einsum('ijkl, mjkl -> imjkl', (rep, output))
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        return output, SstackT

    def forward(self, x): # x is mini_batch_size by input_dim

        phi = F.softplus(self.parameter)

        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)


        """ draw Gamma RVs using phi and 1 """
        num_samps = self.num_samps_for_switch
        concentration_param = phi.view(-1,1).repeat(1,num_samps)
        beta_param = torch.ones(concentration_param.size())
        #Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample() #200, 150, input_dim x samples_num

        if any(torch.sum(gamma_samps,0)==0):
            print("sum of gamma samps are zero!")
        else:
            Sstack = gamma_samps / torch.sum(gamma_samps, 0) # input dim by  # samples

        #x_samps = torch.einsum("ij,jk -> ijk",(x, Sstack)) #([100, 29, 150]) 100- batch size, 150 - samples

        SstackT = Sstack.t()

        output, Sprime = self.switch_func_fc(x, SstackT)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)

        output = output.reshape(mini_batch_size, self.num_samps_for_switch, -1)
        output = torch.mean(output, 1)

        return output, phi









        x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W))) #[100,150]
        labelstack = torch.sigmoid(x_out)

        return labelstack, phi

# def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate):

    if method=="vips":
        BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

        return BCE
    elif method=="nn":
        loss=nn.CrossEntropyLoss()
        BCE = loss(prediction, true_y) #binary

        # KLD term
        alpha_0 = torch.Tensor([alpha_0])
        hidden_dim = torch.Tensor([hidden_dim])

        trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
        trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
        trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))

        KLD = trm1 + trm2 + trm3
        # annealing kl-divergence term is better

        KLD=0 #just to check

        return BCE + annealing_rate*KLD/how_many_samps


def shuffle_data(y,x,how_many_samps):
    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]
    return shuffled_y, shuffled_x


def main():



    """ load pre-trained models """
    # LR_sigma0 = np.load('LR_model0.npy')
    # LR_sigma1 = np.load('LR_model1.npy')
    # LR_sigma10 = np.load('LR_model10.npy')
    # LR_sigma50 = np.load('LR_model50.npy')
    # LR_sigma100 = np.load('LR_model100.npy')

    if dataset == "cervical":
        X_train, y_train, X_test, y_test = load_cervical()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])

    elif dataset == "credit":
        X_train, y_train, X_test, y_test = load_credit()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])
    elif dataset == "adult":
        filename = 'adult.p'
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            y_tot, x_tot = data
    elif dataset == "xor":
        xor_dataset = np.load('../data/synthetic/XOR/dataset_XOR.npy')
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
    elif dataset == "orange_skin":
        xor_dataset = np.load('../data/synthetic/orange_skin/dataset_orange_skin.npy')
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    # unpack data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)

    X = x_tot[:N, :]
    y = y_tot[:N]

    input_dim = d
    hidden_dim = input_dim
    how_many_samps = N

    # preparing variational inference
    alpha_0 = 0.01 # below 1 so that we encourage sparsity.
    num_samps_for_switch = 150

    num_repeat = 5
    # iter_sigmas = np.array([0., 1., 10., 50., 100.])
    iter_sigmas = np.array([0.])




    for k in range(iter_sigmas.shape[0]):
        LR_model = np.load('models/%s_%s_LR_model' % (dataset, method)+str(int(iter_sigmas[k]))+'.npy')
        filename = 'weights/%s_switch_posterior_mean' % dataset+str(int(iter_sigmas[k]))
        filename_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
        posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
        switch_parameter_mat = np.empty([num_repeat, input_dim])

        for repeat_idx in range(num_repeat):
            print(repeat_idx)
            if method=="vips":
                model = Model(input_dim=input_dim, LR_model=torch.Tensor(LR_model[repeat_idx,:]), num_samps_for_switch=num_samps_for_switch)
            elif method=="nn":
                model = Modelnn(d,2, num_samps_for_switch)
                model.load_state_dict(LR_model[()][repeat_idx], strict=False)

            h = model.fc1.weight.register_hook(lambda grad: grad * 0)
            h = model.fc2.weight.register_hook(lambda grad: grad * 0)
            h = model.fc3.weight.register_hook(lambda grad: grad * 0)
            h = model.fc4.weight.register_hook(lambda grad: grad * 0)
            h = model.fc1.bias.register_hook(lambda grad: grad * 0)
            h = model.fc2.bias.register_hook(lambda grad: grad * 0)
            h = model.fc3.bias.register_hook(lambda grad: grad * 0)
            h = model.fc4.bias.register_hook(lambda grad: grad * 0)

            # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            optimizer = optim.Adam(model.parameters(), lr=1e-1)
            how_many_epochs = 10 #150
            how_many_iter = np.int(how_many_samps/mini_batch_size)



            training_loss_per_epoch = np.zeros(how_many_epochs)

            annealing_steps = float(8000.*how_many_epochs)
            beta_func = lambda s: min(s, annealing_steps) / annealing_steps

            ############################################################################################3

            print('Starting Training')

            # for name,par in model.named_parameters():
            #     print (name)


            for epoch in range(how_many_epochs):  # loop over the dataset multiple times

                running_loss = 0.0

                yTrain, xTrain= shuffle_data(y, X, how_many_samps)
                annealing_rate = beta_func(epoch)

                for i in range(how_many_iter):

                    # get the inputs
                    inputs = xTrain[i*mini_batch_size:(i+1)*mini_batch_size,:]
                    labels = yTrain[i*mini_batch_size:(i+1)*mini_batch_size]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs, phi_cand = model(torch.Tensor(inputs)) #100,10,150
                    labels = torch.squeeze(torch.tensor(labels))

                    if method=="vips":
                        loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate)
                    elif method == "nn":
                        loss = loss_function(outputs, labels, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate)

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

            print('Finished Training')

            estimated_params = list(model.parameters())

            """ posterior mean over the switches """
            # num_samps_for_switch
            phi_est = F.softplus(torch.Tensor(estimated_params[0]))

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
            print('estimated parameters are ', phi_est.detach().numpy())

        print(filename, filename_phi)
        np.save(filename,posterior_mean_switch_mat)
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
