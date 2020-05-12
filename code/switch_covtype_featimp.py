"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
from torchvision import datasets, transforms

from switch_model_wrapper import SwitchWrapper, loss_function
from train_covtype_model import CovtypeNet, get_covtype_dataloaders



def load_models_covtype(dataset, selected_label):
  """

  :return: list of (sigma, model generator) pairs
  """
  nn_model = CovtypeNet(selected_label)
  nn_model.load_state_dict(torch.load(f'models/{dataset}_nn_ep7.pt'))

  return [(0, [nn_model])]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200)
  parser.add_argument('--test-batch-size', type=int, default=100)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--selected-label', type=int, default=4)  # label for 1-v-rest training
  parser.add_argument('--n-switch-samples', type=int, default=40)
  parser.add_argument('--dataset', type=str, default='covtype')
  parser.add_argument('--save-model', action='store_true', default=False)
  return parser.parse_args()


def main():

  ar = parse_args()
  assert ar.dataset == 'covtype'
  use_cuda = not ar.no_cuda and torch.cuda.is_available()

  torch.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  train_loader, _ = get_covtype_dataloaders(use_cuda, ar.batch_size, ar.test_batch_size)
  # unpack data
  print(len(train_loader.dataset))
  n_data, n_features = len(train_loader.dataset), 54

  # preparing variational inference
  alpha_0 = 0.01  # below 1 so that we encourage sparsity.
  num_repeat = 1

  classifiers_list = load_models_covtype(ar.dataset, ar.selected_label)

  for sigma, classifiers_gen in classifiers_list:
    posterior_mean_switch_mat = np.empty([num_repeat, n_features])
    switch_parameter_mat = np.empty([num_repeat, n_features])

    for repeat_idx, classifier in enumerate(classifiers_gen):

      model = SwitchWrapper(classifier, n_features, ar.n_switch_samples)
      optimizer = optim.Adam(model.parameters(recurse=False), lr=ar.lr)

      training_loss_per_epoch = np.zeros(ar.epochs)

      annealing_steps = float(8000. * ar.epochs)

      def beta_func(s):
        return min(s, annealing_steps) / annealing_steps

      for epoch in range(ar.epochs):  # loop over the dataset multiple times
        print(f'epoch {epoch}')
        running_loss = 0.0

        annealing_rate = beta_func(epoch)

        for x_batch, y_batch in train_loader:
          x_batch = x_batch.reshape(x_batch.shape[0], -1)

          # zero the parameter gradients
          optimizer.zero_grad()
          # print(y_batch)
          y_batch = (y_batch == ar.selected_label).to(torch.float32)
          # print(y_batch)

          # forward + backward + optimize
          outputs, phi_cand = model(x_batch)  # 100,10,150
          labels = y_batch[:, None].repeat(1, ar.n_switch_samples)
          loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate)
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
      # std_s = torch.std(s_stack, 1)
      posterior_mean_switch = avg_s.detach().numpy()
      # posterior_std_switch = std_s.detach().numpy()

      posterior_mean_switch_mat[repeat_idx, :] = posterior_mean_switch
      print('estimated posterior mean of Switch is', posterior_mean_switch)
      print('estimated parameters are ', phi_est.detach().numpy())

    # save a visualization of the posterior mean switches
    print(posterior_mean_switch_mat.shape)
    # vis_save_file = f'weights/{ar.dataset}_switch_vis_sig{int(sigma)}_label{ar.selected_label}'
    # plot_switches(posterior_mean_switch_mat, vis_save_file)

    # save_file = 'weights/%s_switch_posterior_mean' % dataset + str(int(iter_sigmas[k]))
    # save_file_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
    save_file = f'weights/{ar.dataset}_switch_posterior_mean_sig{int(sigma)}_label{ar.selected_label}'
    save_file_phi = f'weights/{ar.dataset}_switch_parameter_sig{int(sigma)}_label{ar.selected_label}'
    np.save(save_file, posterior_mean_switch_mat)
    np.save(save_file_phi, switch_parameter_mat)


if __name__ == '__main__':
  main()


