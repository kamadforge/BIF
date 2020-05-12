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

from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_mnist_data(use_cuda, batch_size, test_batch_size):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = datasets.MNIST('../data', train=False, transform=transforms.Compose([transform]))
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def load_models_mnist(dataset, selected_label):
  """

  :return: list of (sigma, model generator) pairs
  """
  assert dataset == 'mnist'
  nn_model = MnistNet(selected_label)
  nn_model.load_state_dict(torch.load(f'models/{dataset}_nn_ep4.pt'))

  return [(0, [nn_model])]


def plot_switches(switch_mat, n_rows, n_cols, save_path):
  # normalize to fit the plot
  switch_mat = switch_mat - np.min(switch_mat, axis=1)[:, None]
  switch_mat = switch_mat / np.max(switch_mat, axis=1)[:, None]
  print(np.min(switch_mat), np.max(switch_mat))

  bs = switch_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(switch_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows * i:n_rows * (i + 1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  plt.imsave(save_path + '.png', mnist_mat_flat, cmap=cm.gray, vmin=0., vmax=1.)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--dataset', type=str, default='mnist')
  parser.add_argument('--selected-label', type=int, default=3)  # label for 1-v-rest training
  # parser.add_argument('--log-interval', type=int, default=500)
  parser.add_argument('--n-switch-samples', type=int, default=10)
  parser.add_argument('--alpha_0', type=int, default=0.1)

  parser.add_argument('--save-model', action='store_true', default=False)
  parser.add_argument("--point_estimate", default=False)
  parser.add_argument("--KL_reg", default=False)

  return parser.parse_args()


def main():

  ar = parse_args()
  use_cuda = not ar.no_cuda and torch.cuda.is_available()

  torch.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  # unpack data
  n_data, n_features = 60000, 784

  # preparing variational inference
  # alpha_0 = 0.01  # below 1 so that we encourage sparsity.
  alpha_0 = ar.alpha_0
  num_repeat = 1

  classifiers_list = load_models_mnist(ar.dataset, ar.selected_label)

  for sigma, classifiers_gen in classifiers_list:
    posterior_mean_switch_mat = np.empty([num_repeat, n_features])
    switch_parameter_mat = np.empty([num_repeat, n_features])

    for repeat_idx, classifier in enumerate(classifiers_gen):
      print(repeat_idx)

      model = SwitchWrapper(classifier, n_features, ar.n_switch_samples, ar.point_estimate)
      optimizer = optim.Adam(model.parameters(recurse=False), lr=ar.lr)
      # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
          if ar.point_estimate==False:
            labels = y_batch[:, None].repeat(1, ar.n_switch_samples)
          else:
            labels = y_batch
          loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, ar.KL_reg)
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

      if ar.point_estimate:

        posterior_mean_switch = phi_est / torch.sum(phi_est)
        posterior_mean_switch = posterior_mean_switch.detach().numpy()
        sorted_switch = np.sort(posterior_mean_switch)
        print('estimated switches for the top five important input pixels', np.flip(sorted_switch[-5:]))

      else:

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
        sorted_switch = np.sort(posterior_mean_switch)
        print('estimated switches for the top five important input pixels', np.flip(sorted_switch[-5:]))

      posterior_mean_switch_mat[repeat_idx, :] = posterior_mean_switch
      print('estimated posterior mean of Switch is', posterior_mean_switch)
      print('estimated parameters are ', phi_est.detach().numpy())

    # save a visualization of the posterior mean switches
    print(posterior_mean_switch_mat.shape)
    vis_save_file = f'weights/{ar.dataset}_switch_vis_sig{int(sigma)}'
    plot_switches(posterior_mean_switch_mat, posterior_mean_switch_mat.shape[0], 1, vis_save_file)

    # save_file = 'weights/%s_switch_posterior_mean' % dataset + str(int(iter_sigmas[k]))
    # save_file_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
    save_file = f'weights/{ar.dataset}_switch_posterior_mean_sig{int(sigma)}'
    save_file_phi = f'weights/{ar.dataset}_switch_parameter_sig{int(sigma)}'
    np.save(save_file, posterior_mean_switch_mat)
    np.save(save_file_phi, switch_parameter_mat)


if __name__ == '__main__':
  main()


