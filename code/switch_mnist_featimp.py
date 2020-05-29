"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'

import argparse
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import sys
import torch as pt
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Gamma
# from data.tab_dataloader import load_cervical, load_adult, load_credit
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# from switch_model_wrapper import SwitchWrapper, loss_function, MnistNet
from switch_model_wrapper import SwitchNetWrapper, BinarizedMnistNet, MnistPatchSelector
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class BinarizedMnistDataset(Dataset):
  def __init__(self, train, label_a=3, label_b=8, data_path='../data'):
    super(BinarizedMnistDataset, self).__init__()
    self.train = train
    base_data = datasets.MNIST(data_path, train=train, download=True)

    ids_a, ids_b = base_data.targets == label_a, base_data.targets == label_b
    smp_a, smp_b = base_data.data[ids_a], base_data.data[ids_b]
    n_a, n_b = smp_a.shape[0], smp_b.shape[0]
    print(n_a, n_b)
    tgt_a, tgt_b = base_data.targets[ids_a], base_data.targets[ids_b]
    pert = np.random.permutation(n_a + n_b)
    tgt = np.concatenate([np.zeros(tgt_a.shape), np.ones(tgt_b.shape)])[pert]
    smp = np.concatenate([smp_a, smp_b])[pert]

    smp = np.reshape(smp, (-1, 784))
    self.tgt = tgt.astype(np.float32)
    self.smp = smp.astype(np.float32) / 255

  def __len__(self):
    return self.tgt.shape[0]

  def __getitem__(self, idx):
    return pt.tensor(self.smp[idx]), pt.tensor(self.tgt[idx])


def load_mnist_data(use_cuda, batch_size, test_batch_size, data_path='../data', ):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
  train_loader = pt.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([transform]))
  test_loader = pt.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def load_two_label_mnist_data(use_cuda, batch_size, test_batch_size, data_path='../data', label_a=3, label_b=8):
  # select only samples of labels a and b, then binarize labels as a=0 and b=1

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_data = BinarizedMnistDataset(train=True, label_a=label_a, label_b=label_b, data_path=data_path)
  test_data = BinarizedMnistDataset(train=False, label_a=label_a, label_b=label_b, data_path=data_path)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def switch_select_data(selector, loader, device):
    x_data, y_data, selection = [], [], []
    with pt.no_grad():
      for x_tr, y_tr in loader:
        x_data.append(x_tr.numpy())
        y_data.append(y_tr.numpy())
        x_sel = nnf.softplus(selector(x_tr.to(device)))
        x_sel = x_sel / pt.sum(x_sel, dim=1)[:, None] * 16  # multiply by patch size
        selection.append(x_sel.cpu().numpy())
    return np.concatenate(x_data), np.concatenate(y_data), np.concatenate(selection)


def hard_select_data(data, selection, mode='topk', k=1, baseline_val=0):
  # if mode == 'mult':
  #   return data * selection
  if mode.startswith('topk'):
    # due to 4x4 patching, each value occurs 16 times
    effective_k = 16 * k
    sorted_selection = np.sort(selection, axis=1)
    threshold_by_sample = sorted_selection[:, -effective_k]

    below_threshold = selection < threshold_by_sample[:, None]  # , 784, axis=1)
    # print(below_threshold[0])
    feats_selected = 784 - np.sum(below_threshold, axis=1)
    assert np.max(feats_selected) == float(effective_k)
    assert np.max(feats_selected) == np.min(feats_selected)
    data_to_select = np.copy(data)
    data_to_select[below_threshold] = baseline_val

    # if mode == 'topk_mult':
    #   data_to_select = data_to_select * selection

    # make sure no sample has more nonzero entries than allowed in the selection
    assert np.max(np.sum(data_to_select != baseline_val, axis=1)) <= float(effective_k)
    return data_to_select
  else:
    raise ValueError


def make_select_loader(x_select, y_select, train, batch_size, use_cuda):
  train_data = BinarizedMnistDataset(train=train)
  train_data.smp, train_data.tgt = x_select, y_select
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  return data_loader


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


def test_classifier_epoch(classifier, test_loader, device):
  classifier.eval()
  test_loss = 0
  correct = 0
  with pt.no_grad():
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      pred = classifier(x_batch)
      test_loss = nnf.binary_cross_entropy_with_logits(pred, y_batch, reduction='sum').item()
      # test_loss = nnf.nll_loss(pred, y_batch, reduction='sum').item()  # sum up batch loss
      class_pred = pt.sigmoid(pred).round()  # get the index of the max log-probability
      # print(class_pred.shape, y_batch.shape)
      correct += class_pred.eq(y_batch.view_as(class_pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def train_classifier(classifier, train_loader, test_loader, epochs, lr, device):
  optimizer = optim.Adam(classifier.parameters(), lr=lr)

  for epoch in range(epochs):  # loop over the dataset multiple times
    classifier.train()
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()

      loss = nnf.binary_cross_entropy_with_logits(classifier(x_batch), y_batch)
      # loss = loss_function(outputs, labels, phi_cand, alpha_0, n_features, n_data, annealing_rate, KL_reg)
      loss.backward()
      optimizer.step()

    test_classifier_epoch(classifier, test_loader, device)


def train_selector(model, train_loader, epochs, lr, device):
  # , point_estimate, n_switch_samples, alpha_0, n_features, n_data, KL_reg):
  optimizer = optim.Adam(model.selector_params(), lr=lr)
  training_loss_per_epoch = np.zeros(epochs)
  # annealing_steps = float(8000. * epochs)

  # def beta_func(s):
  #   return min(s, annealing_steps) / annealing_steps

  for epoch in range(epochs):  # loop over the dataset multiple times
    print(f'epoch {epoch}')
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
    print('epoch number is ', epoch)
    print('running loss is ', running_loss)


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
  test_classifier_epoch(classifier, select_test_loader, device)

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
