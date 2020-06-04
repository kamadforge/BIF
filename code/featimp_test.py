"""
Test learning feature importance under DP and non-DP models
"""

# For instancewise training one can do two things:

# 1. train the switch vector by backpropagation (the original way) and then finetune on one example (local training)
#  set training_local to True
#  set switch_nn to False

# 2. Train the switch network which outputs the switch parameters (phi) and feeds them on the classifier
# set switch_nn to True
# possibly set_hooks to False
# set training_local to False


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
import sys
sys.path.append("/home/kadamczewski/Dropbox_from/Current_research/featimp_dp")
sys.path.append("/home/kadamczewski/Dropbox_from/Current_research/featimp_dp/data")
sys.path.append("/home/kadamczewski/Dropbox_from/Current_research/featimp_dp/code")
sys.path.append("/home/kadamczewski/Dropbox_from/Current_research/featimp_dp/code/models")


from data.synthetic_data_loader import synthetic_data_loader
from evaluation_metrics import compute_median_rank, binary_classification_metrics

from models.nn_3hidden import FC


########################################
# PATH

cwd = os.getcwd()
cwd_parent = Path(__file__).parent.parent

if socket.gethostname()=='worona.local':
    pathmain = cwd
    path_code = os.path.join(pathmain, "code")
elif socket.gethostname()=='kamil':
    pathmain=cwd_parent
    path_code=cwd
#if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
else:
    sys.path.append(os.path.join(cwd_parent, "data"))
    pathmain=cwd
    path_code = os.path.join(pathmain, "code")
    #path_code = os.path.join(pathmain)


##################################################3
# ARGUMENTS


def get_args():

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--dataset", default="adult_short") #xor, orange_skin, nonlinear, alternating, syn4, syn5, syn6
    parser.add_argument("--method", default="nn")
    parser.add_argument("--mini_batch_size", default=110, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.1, type=float)

    # for switch training
    parser.add_argument("--num_Dir_samples", default=50, type=int)
    parser.add_argument("--alpha", default=5, type=float)
    parser.add_argument("--point_estimate", default=True)

    parser.add_argument("--train", default=True)
    parser.add_argument("--test", default=True)

    # for instance wise training
    parser.add_argument("--switch_nn", default=True)

    parser.add_argument("--training_local", default=False)
    parser.add_argument("--local_training_iter", default=200, type=int)
    parser.add_argument("--set_hooks", default=True)
    parser.add_argument("--kl_term", default=False)

    args = parser.parse_args()

    return args

args = get_args()
print(args)




#######################
# LOSS



def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method, kl_term, point_estimate):

    if method=="vips":
        BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

        return BCE
    elif method=="nn":

        if args.switch_nn:

            loss = nn.CrossEntropyLoss()

            if point_estimate:
                if args.mini_batch_size == 1:
                    true_y = true_y.unsqueeze(0)
                BCE = loss(prediction, true_y)

            else: #sampling

                BCE_mat = torch.zeros(prediction.shape[1])
                for ind in torch.arange(0, prediction.shape[1]):
                    y_pred = prediction[:, ind, :]
                    BCE_mat[ind] = loss(y_pred, true_y)

                BCE = torch.mean(BCE_mat)


            if kl_term:

                if point_estimate:
                    # KLD termaa
                    alpha_0 = torch.Tensor([alpha_0])
                    hidden_dim = torch.Tensor([hidden_dim])

                    trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
                    trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
                    trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))

                    KLD = trm1 + trm2 + trm3

                    return BCE + + annealing_rate*KLD/how_many_samps

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

                    # print('KLD and BCE', [KLD/mini_batch_size, BCE])

                    # return BCE + + annealing_rate * KLD / how_many_samps
                    return BCE + KLD / mini_batch_size

            else: #no kl term (both point estimate and sampling)

                return BCE

        else: #non-switch nn

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


                # KLD termaa
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

#######################################################


def main():


    dataset = args.dataset
    method = args.method
    mini_batch_size = args.mini_batch_size
    point_estimate = args.point_estimate

    if method == "nn":
        if args.switch_nn:
            from models.switch_MLP import Model_switchlearning
        else:
            from models.switch_MLP import Modelnn

    elif method=="vips":
        from models.switch_LR import Model


    ###########################################33
    # LOAD DATA

    x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset)



    # unpack data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)

    if dataset == "adult_short":
        N = 26048
    elif dataset == "credit":
        N = 2668

    X = x_tot[:N, :]
    y = y_tot[:N]




    if dataset == "alternating" or "syn" in dataset:
        datatypes = datatypes_tot[:N] #only for alternating, if datatype comes from orange_skin or nonlinear
    else:
        datatypes = None

    X_test = x_tot[N:, :]
    y_test = y_tot[N:]
    if dataset == "alternating" or "syn" in dataset:
        datatypes_test = datatypes_tot[N:]

    input_dim = d
    hidden_dim = input_dim
    how_many_samps = N

    #######################################################
    # preparing variational inference to learn switch vector

    alpha_0 = args.alpha #0.01 # below 1 so that we encourage sparsity. #dirichlet dist parameters
    num_samps_for_switch = args.num_Dir_samples
    num_repeat = 1 # repeating the entire experiment

    # noise
    # iter_sigmas = np.array([0., 1., 10., 50., 100.])
    iter_sigmas = np.array([0.])

    if args.train:

        for k in range(iter_sigmas.shape[0]):

            #load pretrained model
            LR_model = np.load(os.path.join(path_code, 'models/%s_%s_LR_model' % (dataset, method) + str(int(iter_sigmas[k])) + '.npy'), allow_pickle=True)

            filename = os.path.join(path_code, 'weights/%s_%d_%.1f_%d_switch_posterior_mean' % (dataset, args.num_Dir_samples, args.alpha, args.epochs)+str(int(iter_sigmas[k])))
            filename_last = os.path.join(path_code, 'weights/%s_switch_posterior_mean' % (dataset)+str(int(iter_sigmas[k])))
            filename_phi = os.path.join(path_code, 'weights/%s_%d_%.1f_%d_switch_parameter' % (dataset, args.num_Dir_samples, args.alpha, args.epochs)+ str(int(iter_sigmas[k])))

            posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
            switch_parameter_mat = np.empty([num_repeat, input_dim])

            mean_of_means=np.zeros(input_dim)


        ############################################3
        # TRAINING

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

                    # hooks to not update other parameters than switch-related
                    if args.set_hooks:
                        # in case you use pre-trained classifier


                        h = model.fc1.weight.register_hook(lambda grad: grad * 0)
                        h = model.fc2.weight.register_hook(lambda grad: grad * 0)
                        h = model.fc4.weight.register_hook(lambda grad: grad * 0)
                        #h = model.bn1.weight.register_hook(lambda grad: grad * 0)
                        #h = model.bn2.weight.register_hook(lambda grad: grad * 0)

                        h = model.fc1.bias.register_hook(lambda grad: grad * 0)
                        h = model.fc2.bias.register_hook(lambda grad: grad * 0)
                        h = model.fc4.bias.register_hook(lambda grad: grad * 0)
                        #h = model.bn1.bias.register_hook(lambda grad: grad * 0)
                        #h = model.bn2.bias.register_hook(lambda grad: grad * 0)

                ############################################################################################3

                print('Starting Training')

                #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                #optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)

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
                        if (args.dataset == "alternating" or "syn" in args.dataset):
                            datatypes_train_batch = datatypesTrain[i*mini_batch_size:(i+1)*mini_batch_size]
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        outputs, phi_cand, S, prephi = model(torch.Tensor(inputs), mini_batch_size) #100,10,150

                        if method=="vips":
                            labels = torch.squeeze(torch.Tensor(labels))
                            loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method)

                        elif method == "nn":
                            labels = torch.squeeze(torch.LongTensor(labels))
                            loss = loss_function(outputs, labels, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate, method, args.kl_term, point_estimate)

                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                        if i % how_many_iter ==0:
                        #if i % how_many_iter == 0 or i % how_many_iter == 1 or i % how_many_iter == 2:
                            #print(datatypes_train_batch[0:2])
                            if args.switch_nn: #instancewise, averaging over all the exmaples
                                if point_estimate:
                                    #print(S[0:2])
                                    print("switch batch mean: ", S.mean(dim=0))
                                else:
                                    print(S.mean(dim=0).mean(dim=1))  # batch x feat x samtc
                            else: #non switchnn
                                if point_estimate:
                                    print(S)
                                else:
                                    print(S.mean(dim=0))
                            # phis = phi_cand / torch.sum(phi_cand)
                            # print("switch: ", phis.mean(dim=0))
                            # print("switch: ", phis[1:4])
                            #if not point_estimate:
                            #    print("mean of switch samples: ", torch.mean(S[0,:,:],1))

                    # training_loss_per_epoch[epoch] = running_loss/how_many_samps

                    training_loss_per_epoch[epoch] = running_loss
                    print('epoch number is ', epoch)
                    print('running loss is \n', running_loss)

                print('Finished global Training')



                ###################################################

                if args.training_local and dataset == "alternating":

                    print("\nStarting local training")

                    how_many_iter = args.local_training_iter

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

                    torch.save(model.state_dict(),
                               os.path.join(path_code,
                                            f"models/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt"))

                    ###########3


                else: #if switch_nn is true testing a single instance

                    torch.save(model.state_dict(),
                               os.path.join(path_code,
                                            f"models/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt"))

        ########################


        print(filename, filename_phi)
        np.save(filename, posterior_mean_switch_mat)
        np.save(filename_last, posterior_mean_switch_mat)
        np.save(filename_phi, switch_parameter_mat)

###################################################################################################

    if args.test:

            #############################
            # running the test

            def test_instance(dataset, switch_nn, training_local):

                print(f"dataset: {dataset}")

                path = os.path.join(path_code, f"models/switches_{args.dataset}_batch_{args.mini_batch_size}_lr_{args.lr}_epochs_{args.epochs}.pt")

                i = 0  # choose a sample
                mini_batch_size = 10000
                datatypes_test_samp=None


                if switch_nn:
                    model = Model_switchlearning(d,2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)
                else:
                    model = Modelnn(d,2, num_samps_for_switch, mini_batch_size, point_estimate=point_estimate)


                model.load_state_dict(torch.load(path), strict=False)


                inputs_test_samp = X_test[i * mini_batch_size:(i + 1) * mini_batch_size, :] #(mini_batch_size* feat_num)
                labels_test_samp = y_test[i * mini_batch_size:(i + 1) * mini_batch_size]
                if dataset == "alternating" or "syn" in dataset:
                    datatypes_test_samp = datatypes_test[i * mini_batch_size:(i + 1) * mini_batch_size]

                if "syn" in dataset:
                    relevant_features=[]
                    for i in range(datatypes_test_samp.shape[0]):
                        relevant_features.append(np.where(datatypes_test_samp[i]>0))
                    datatypes_test_samp = np.array(relevant_features).squeeze(1)


                inputs_test_samp = torch.Tensor(inputs_test_samp)

                model.eval()

                outputs, phi, S, phi_est = model(inputs_test_samp, mini_batch_size)
                torch.set_printoptions(profile="full")

                #####################
                #########################3
                ##############################
                inputs_test_samp1=inputs_test_samp.clone()
                inputs_test_samp3=inputs_test_samp.clone()
                inputs_test_samp5=inputs_test_samp.clone()

                #######################
                ##########################



                k=1
                #local samples results:
                instance_best_features_ascending = np.argsort(S.detach().cpu().numpy(), axis=1)
                instance_unimportant_features = instance_best_features_ascending[:, :-k]
                #np.save(os.path.join(path_code, str(f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy"), instance_unimportant_features))


                #########################

                #unimportant_features_instance = np.load(f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy")

                for i, data in enumerate(inputs_test_samp1):
                    inputs_test_samp1[i, instance_unimportant_features[i]] = 0


                ###################################
                i = 0  # choose a sample
                mini_batch_size = 2000
                datatypes_test_samp = None

                input_num= d
                output_num = 2

                model = FC(input_num, output_num)

                LR_model = np.load(os.path.join(path_code, 'models/%s_%s_LR_model0.npy' % (dataset,method)), allow_pickle=True)

                model.load_state_dict(LR_model[()][0], strict=False)





                y_pred = model(torch.Tensor(inputs_test_samp1))
                y_pred = torch.argmax(y_pred, dim=1)

                accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == y_test) / len(y_test))
                print("test accuracy 1: ", accuracy)



                ##############################3
                ################################3




                k=3
                #local samples results:
                instance_best_features_ascending = np.argsort(S.detach().cpu().numpy(), axis=1)
                instance_unimportant_features = instance_best_features_ascending[:, :-k]
                #np.save(os.path.join(path_code, f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy", instance_unimportant_features))


                #########################

                #unimportant_features_instance = np.load(f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy")

                for i, data in enumerate(inputs_test_samp3):
                    inputs_test_samp3[i, instance_unimportant_features[i]] = 0


                ###################################
                i = 0  # choose a sample
                mini_batch_size = 2000
                datatypes_test_samp = None

                input_num= d
                output_num = 2

                model = FC(input_num, output_num)

                LR_model = np.load(os.path.join(path_code, 'models/%s_%s_LR_model0.npy' % (dataset,method)), allow_pickle=True)

                model.load_state_dict(LR_model[()][0], strict=False)





                y_pred = model(torch.Tensor(inputs_test_samp3))
                y_pred = torch.argmax(y_pred, dim=1)

                accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == y_test) / len(y_test))
                print("test accuracy 3: ", accuracy)

                ##############################3
                ##################################


                k=5
                #local samples results:
                instance_best_features_ascending = np.argsort(S.detach().cpu().numpy(), axis=1)
                instance_unimportant_features = instance_best_features_ascending[:, :-k]
                #np.save(os.path.join(path_code, f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy", instance_unimportant_features))


                #########################

                #unimportant_features_instance = np.load(f"rankings/instance_featureranks_test_qfit_{dataset}_k_{k}.npy")

                for i, data in enumerate(inputs_test_samp5):
                    inputs_test_samp5[i, instance_unimportant_features[i]] = 0


                ###################################
                i = 0  # choose a sample
                mini_batch_size = 2000
                datatypes_test_samp = None

                input_num= d
                output_num = 2

                model = FC(input_num, output_num)

                LR_model = np.load(os.path.join(path_code, 'models/%s_%s_LR_model0.npy' % (dataset,method)), allow_pickle=True)

                model.load_state_dict(LR_model[()][0], strict=False)





                y_pred = model(torch.Tensor(inputs_test_samp5))
                y_pred = torch.argmax(y_pred, dim=1)

                accuracy = (np.sum(np.round(y_pred.detach().cpu().numpy().flatten()) == y_test) / len(y_test))
                print("test accuracy 5: ", accuracy)
                ##############################
                ######################3
                #####################


                samples_to_see=5
                if mini_batch_size>samples_to_see and datatypes_test_samp is not None:
                    print(datatypes_test_samp[:samples_to_see])
                    print("outputs", outputs[:samples_to_see])
                    print("phi", phi[:samples_to_see])
                return S, datatypes_test_samp


            dataset=args.dataset
            S, datatypes_test_samp = test_instance(dataset, args.switch_nn, False)
            if dataset=="xor":
                k=2
            elif dataset == "orange_skin" or dataset == "nonlinear_additive":
                k=4
            elif dataset == "alternating":
                k=5
            elif dataset == "syn4":
                k=7
            elif dataset == "syn5" or dataset == "syn6":
                k=9

            #######################################
            # evaluation

            synthetic=["xor", "orange_skin", "nonlinear_additive", "alternating", "syn4", "syn5", "syn6"]

            if (args.dataset in synthetic) and (args.switch_nn) :
                if not args.point_estimate:
                    S=S.mean(dim=2)

                median_ranks = compute_median_rank(S, k, dataset, datatypes_test_samp)
                mean_median_ranks=np.mean(median_ranks)
                #if not args.point_estimate:
                #    S=S.mean(dim=1)
                tpr, fdr = binary_classification_metrics(S, k, dataset, mini_batch_size, datatypes_test_samp)
                print("mean median rank", mean_median_ranks)
                print(f"tpr: {tpr}, fdr: {fdr}")

            else:
                tpr, fdr = -1,-1


    return tpr, fdr, S


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
    runs = 1

    tprs, fdrs, Ss = [], [], []
    for i in range(runs):
        print(f"\n\nRun: {i}\n")
        tpr, fdr, S = main()
        tprs.append(tpr); fdrs.append(fdr); Ss.append(S.mean(dim=0).detach().numpy())

    print("*"*50)
    S_average = np.round(np.mean(Ss, axis=0),3)
    S_average_nums = np.argsort(S_average)[::-1]
    print(f"tpr mean {np.mean(tprs)}, fdr mean: {np.mean(fdrs)}, tpr std {np.std(tprs)}, fdr_std {np.std(fdrs)}, S_testmean: {S_average}, S_args: {S_average_nums}")
    print(",".join([str(a) for a in S_average_nums]))
