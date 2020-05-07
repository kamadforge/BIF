"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Gamma
import pickle

from data.tab_dataloader import load_cervical, load_adult, load_credit
from switch_model_wrapper import SwitchWrapper, loss_function, LogReg


def shuffle_data(y,x,how_many_samps):
  idx = np.random.permutation(how_many_samps)
  shuffled_y = y[idx]
  shuffled_x = x[idx,:]
  return shuffled_y, shuffled_x


def load_data(dataset):
  if dataset == "cervical":
    x_train, y_train, x_test, y_test = load_cervical()
    x_total = np.concatenate([x_train, x_test])
    y_total = np.concatenate([y_train, y_test])

  elif dataset == "credit":
    x_train, y_train, x_test, y_test = load_credit()
    x_total = np.concatenate([x_train, x_test])
    y_total = np.concatenate([y_train, y_test])
  elif dataset == "adult":
    filename = 'adult.p'
    with open(filename, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
      y_total, x_total = data
  elif dataset == "xor":
    xor_dataset = np.load('../data/synthetic/XOR/dataset_XOR.npy')
    x_total = xor_dataset[()]['x']
    y_total = xor_dataset[()]['y']
  elif dataset == "orange_skin":
    xor_dataset = np.load('../data/synthetic/orange_skin/dataset_orange_skin.npy')
    x_total = xor_dataset[()]['x']
    y_total = xor_dataset[()]['y']
  else:
    raise ValueError
  return x_total, y_total


def load_models_logreg(dataset, iter_sigmas):
  """

  :return: list of (sigma, model generator) pairs
  """
  def model_gen_logreg(sigma):
    load_string = f'models/{dataset}_LR_model{str(int(sigma))}.npy'
    lr_weights = np.load(load_string)
    n_repeats, n_features = lr_weights.shape

    for idx in range(n_repeats):
      log_reg_model = LogReg(d_in=n_features, d_out=1)
      log_reg_model.load(torch.tensor(lr_weights[idx, :], dtype=torch.float32)[None, :])
      yield log_reg_model

  return [(sig, model_gen_logreg(sig)) for sig in iter_sigmas]


def main():

  dataset = 'adult'

  """ load pre-trained models """
  # LR_sigma0 = np.load('LR_model0.npy')
  # LR_sigma1 = np.load('LR_model1.npy')
  # LR_sigma10 = np.load('LR_model10.npy')
  # LR_sigma50 = np.load('LR_model50.npy')
  # LR_sigma100 = np.load('LR_model100.npy')

  x_total, y_total = load_data(dataset)
  # unpack data
  n_tot, input_dim = x_total.shape

  training_data_por = 0.8

  how_many_samps = int(training_data_por * n_tot)

  x_data = x_total[:how_many_samps, :]
  y_data = y_total[:how_many_samps]

  hidden_dim = input_dim

  # preparing variational inference
  alpha_0 = 0.01  # below 1 so that we encourage sparsity.
  num_samps_for_switch = 150

  num_repeat = 5
  iter_sigmas = np.array([0., 1., 10., 50., 100.])

  classifiers_list = load_models_logreg(dataset, iter_sigmas)

  for sigma, classifiers_gen in classifiers_list:
    posterior_mean_switch_mat = np.empty([num_repeat, input_dim])
    switch_parameter_mat = np.empty([num_repeat, input_dim])

    for repeat_idx, classifier in enumerate(classifiers_gen):
      print(repeat_idx)

      model = SwitchWrapper(classifier, input_dim, num_samps_for_switch)

      # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
      optimizer = optim.Adam(model.parameters(recurse=False), lr=1e-1)
      mini_batch_size = 100
      how_many_epochs = 20 # 150
      how_many_iter = np.int(how_many_samps/mini_batch_size)

      training_loss_per_epoch = np.zeros(how_many_epochs)

      annealing_steps = float(8000.*how_many_epochs)
      beta_func = lambda s: min(s, annealing_steps) / annealing_steps

      ############################################################################################

      print('Starting Training')

      for name, par in model.named_parameters(recurse=False):
        print(name)

      for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        y_train, x_train = shuffle_data(y_data, x_data, how_many_samps)
        annealing_rate = beta_func(epoch)

        for i in range(how_many_iter):

          # get the inputs
          inputs = x_train[i*mini_batch_size:(i+1)*mini_batch_size,:]
          labels = y_train[i * mini_batch_size:(i + 1) * mini_batch_size]

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs, phi_cand = model(torch.Tensor(inputs))  # 100,10,150
          labels = torch.squeeze(torch.Tensor(labels))
          loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand, alpha_0,
                               hidden_dim, how_many_samps, annealing_rate)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()

        # training_loss_per_epoch[epoch] = running_loss/how_many_samps
        training_loss_per_epoch[epoch] = running_loss
        print('epoch number is ', epoch)
        print('running loss is ', running_loss)

      print('Finished Training')

      estimated_params = list(model.parameters(recurse=False))

      """ posterior mean over the switches """
      # num_samps_for_switch
      phi_est = F.softplus(torch.Tensor(estimated_params[0]))

      switch_parameter_mat[repeat_idx, :] = phi_est.detach().numpy()

      concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
      # beta_param = torch.ones(self.hidden_dim,1).repeat(1,num_samps)
      beta_param = torch.ones(concentration_param.size())
      gamma_obj = Gamma(concentration_param, beta_param)
      gamma_samps = gamma_obj.rsample()
      s_stack = gamma_samps / torch.sum(gamma_samps, 0)
      avg_s = torch.mean(s_stack, 1)
      std_s = torch.std(s_stack, 1)
      posterior_mean_switch = avg_s.detach().numpy()
      posterior_std_switch = std_s.detach().numpy()

      posterior_mean_switch_mat[repeat_idx,:] = posterior_mean_switch
      print('estimated posterior mean of Switch is', posterior_mean_switch)
      print('estimated parameters are ', phi_est.detach().numpy())

    # save_file = 'weights/%s_switch_posterior_mean' % dataset + str(int(iter_sigmas[k]))
    # save_file_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
    save_file = 'weights/%s_switch_posterior_mean' % dataset + str(int(sigma))
    save_file_phi = 'weights/%s_switch_parameter' % dataset + str(int(sigma))
    np.save(save_file, posterior_mean_switch_mat)
    np.save(save_file_phi, switch_parameter_mat)

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
  torch.manual_seed(42)
  np.random.seed(42)
  main()
