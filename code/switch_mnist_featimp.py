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
import torch as pt
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
# from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
from switch_model_wrapper import SwitchNetWrapper, BinarizedMnistNet, MnistPatchSelector
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
from mnist_utils import plot_switches, load_two_label_mnist_data, switch_select_data, hard_select_data, \
  make_select_loader, train_classifier
from mnist_posthoc_accuracy_eval import test_posthoc_acc


def global_switch_eval(model, point_estimate, ar):
  estimated_params = list(model.selector_params())
  phi_est = nnf.softplus(pt.Tensor(estimated_params[0]))

  if point_estimate:
    posterior_mean_switch = phi_est / pt.sum(phi_est)
    posterior_mean_switch = posterior_mean_switch.detach().numpy()

  else:
    switch_parameter_mat = phi_est.detach().numpy()

    concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
    # beta_param = pt.ones(self.hidden_dim,1).repeat(1,num_samps)
    beta_param = pt.ones(concentration_param.size())
    gamma_obj = Gamma(concentration_param, beta_param)
    gamma_samps = gamma_obj.rsample()
    s_stack = gamma_samps / pt.sum(gamma_samps, 0)
    avg_s = pt.mean(s_stack, 1)
    # std_s = pt.std(s_stack, 1)
    posterior_mean_switch = avg_s.detach().numpy()

    save_file_phi = f'weights/{ar.dataset}_switch_parameter '
    np.save(save_file_phi, switch_parameter_mat)

  kl_str = '' if not ar.KL_reg else f'_alpha{ar.alpha_0}'
  vis_save_file = f'weights/{ar.dataset}_switch_vis_label{ar.selected_label}_lr{ar.lr}{kl_str}'
  plot_switches(posterior_mean_switch, posterior_mean_switch.shape[0], 1, vis_save_file)

  # save_file = 'weights/%s_switch_posterior_mean' % dataset + str(int(iter_sigmas[k]))
  # save_file_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
  # save_file = f'weights/{ar.dataset}_switch_posterior_mean'


def train_selector(model, train_loader, epochs, lr, device):
  # , point_estimate, n_switch_samples, alpha_0, n_features, n_data, KL_reg):
  optimizer = optim.Adam(model.selector_params(), lr=lr)
  training_loss_per_epoch = np.zeros(epochs)
  # annealing_steps = float(8000. * epochs)

  # def beta_func(s):
  #   return min(s, annealing_steps) / annealing_steps

  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0

    # annealing_rate = beta_func(epoch)

    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs, phi_cand = model(x_batch)  # 100,10,150

      loss = nnf.binary_cross_entropy_with_logits(outputs, y_batch)
      # loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()

    # training_loss_per_epoch[epoch] = running_loss/how_many_samps
    training_loss_per_epoch[epoch] = running_loss
    print(f'Epoch {epoch} Loss {running_loss}')


def local_switch_eval(model, test_loader):
  pass


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=2)
  parser.add_argument('--dataset', type=str, default='mnist')
  # parser.add_argument('--selected-label', type=int, default=3)  # label for 1-v-rest training
  # parser.add_argument('--log-interval', type=int, default=500)
  parser.add_argument('--n-switch-samples', type=int, default=10)

  parser.add_argument('--save-model', action='store_true', default=False)
  parser.add_argument("--point_estimate", default=True)
  # parser.add_argument("--KL_reg", default=False)
  # parser.add_argument('--alpha_0', type=float, default=50000.)
  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  parser.add_argument('--select-k', type=int, default=4)

  # parser.add_argument("--freeze-classifier", default=True)
  # parser.add_argument("--patch-selection", default=True)

  return parser.parse_args()


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  # train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  train_loader, test_loader = load_two_label_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size,
                                                        label_a=ar.label_a, label_b=ar.label_b)
  # unpack data
  n_data, n_features = 60000, 784

  classifier = BinarizedMnistNet().to(device)
  # classifier.load_state_dict(pt.load(f'models/{ar.dataset}_nn_ep4.pt'))
  train_classifier(classifier, train_loader, test_loader, ar.epochs, ar.lr, device)
  print('Finished Training Classifier')

  selector = MnistPatchSelector().to(device)
  model = SwitchNetWrapper(selector, classifier, n_features, ar.n_switch_samples, ar.point_estimate).to(device)


  train_selector(model, train_loader, ar.epochs, ar.lr, device)
  # , ar.point_estimate, ar.n_switch_samples, ar.alpha_0, n_features, n_data, ar.KL_reg)
  print('Finished Training Selector')



  x_tr, y_tr, tr_selection = switch_select_data(selector, train_loader, device)
  x_ts, y_ts, ts_selection = switch_select_data(selector, test_loader, device)
  x_tr_select = hard_select_data(x_tr, tr_selection, k=ar.select_k)
  x_ts_select = hard_select_data(x_ts, ts_selection, k=ar.select_k)
  select_train_loader = make_select_loader(x_tr_select, y_tr, train=True, batch_size=ar.batch_size, use_cuda=use_cuda)
  select_test_loader = make_select_loader(x_ts_select, y_ts, train=False, batch_size=ar.test_batch_size,
                                          use_cuda=use_cuda)
  print('testing classifier')
  # test_classifier_epoch(classifier, select_test_loader, device)
  test_posthoc_acc(ar.label_a, ar.label_b, select_test_loader, device)
  # print('testing retrained classifier')
  # new_classifier = BinarizedMnistNet().to(device)
  # train_classifier(new_classifier, select_train_loader, select_test_loader, ar.epochs, ar.lr, device)


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  do_featimp_exp(ar)


if __name__ == '__main__':
  main()
