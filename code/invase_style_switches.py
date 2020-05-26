"""Instance-wise Variable Selection (INVASE) module - with baseline

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
           "IINVASE: Instance-wise Variable Selection using Neural Networks,"
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com
"""
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
import argparse
from comparison_methods.INVASE.data_generation import generate_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class Net(nn.Module):
  def __init__(self, d_in, d_hid, d_out, act='relu', act_out='logsoftmax', use_bn=True):
    assert act in {'relu', 'selu'}
    assert act_out in {'logsoftmax', 'softplus'}
    super(Net, self).__init__()
    self.use_bn = use_bn
    self.act = nn.ReLU() if act == 'relu' else nn.SELU()
    self.act_out = nn.LogSoftmax(dim=1) if act_out == 'logsoftmax' else nn.Softplus()
    self.fc1 = nn.Linear(d_in, d_hid)
    self.bn1 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.bn2 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc3 = nn.Linear(d_hid, d_out)

  def forward(self, x_in):
    x = self.fc1(x_in)
    x = self.bn1(x) if self.use_bn else x
    x = self.act(x)
    x = self.fc2(x)
    x = self.bn2(x) if self.use_bn else x
    x = self.act(x)
    x = self.act_out(self.fc3(x))
    return x


class InvaseSwitch(nn.Module):
  """INVASE class.

  Attributes:
    - x_train: training features
    - y_train: training labels
    - model_type: invase or invase_minus
    - model_parameters:
      - actor_h_dim: hidden state dimensions for actor
      - critic_h_dim: hidden state dimensions for critic
      - n_layer: the number of layers
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE
  """

  def __init__(self, x_train, y_train, model_type, model_parameters, device):
    super(InvaseSwitch, self).__init__()

    self.lamda = model_parameters['lamda']
    self.actor_h_dim = model_parameters['actor_h_dim']
    self.critic_h_dim = model_parameters['critic_h_dim']
    self.n_layer = model_parameters['n_layer']
    self.batch_size = model_parameters['batch_size']
    self.iteration = model_parameters['iteration']
    self.activation = model_parameters['activation']
    self.learning_rate = model_parameters['learning_rate']
    self.alpha_0 = model_parameters['alpha_0']
    self.n_data = model_parameters['n_data']
    self.kl_weight = model_parameters['kl_weight']


    self.device = device
    self.dim = x_train.shape[1]
    self.label_dim = y_train.shape[1]

    self.model_type = model_type

    # Build and compile critic
    self.critic_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                          act_out='logsoftmax').to(self.device)
    # self.critic.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # Build and compile the actor
    self.actor_net = Net(self.dim, self.actor_h_dim, self.dim, act=self.activation,
                         act_out='softplus', use_bn=False).to(self.device)
    # self.actor.compile(loss=self.actor_loss, optimizer=optimizer)

    self.total_parameters = list(self.critic_net.parameters()) + list(self.actor_net.parameters())

    if self.model_type in {'invase', 'mse_match', 'ce_match'}:
      # Build and compile the baseline
      self.baseline_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                              act_out='logsoftmax').to(self.device)
      # self.baseline.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
      if self.model_type == 'invase':
        self.total_parameters.extend(list(self.baseline_net.parameters()))

  def invase_loss(self, selection, log_critic_out, log_baseline_out, y_batch_onehot, y_batch_scalar, actor_out):
    selection = selection.detach()
    critic_loss = -pt.sum(y_batch_onehot * log_critic_out.detach(), dim=1)

    combined_loss = nnf.nll_loss(log_critic_out, y_batch_scalar)

    if self.model_type == 'invase':
      # Baseline loss
      baseline_loss = -pt.sum(y_batch_onehot * log_baseline_out.detach(), dim=1)
      combined_loss += nnf.nll_loss(log_baseline_out, y_batch_scalar)
      # Reward
      Reward = -(critic_loss - baseline_loss)
    elif self.model_type == 'invase_minus':
      Reward = -critic_loss

    else:
      raise ValueError

    # Policy gradient loss computation.
    actor_term = pt.sum(selection * pt.log(actor_out + 1e-8) + (1 - selection) * pt.log(1 - actor_out + 1e-8), dim=1)
    sparcity_term = pt.mean(actor_out, dim=1)
    custom_actor_loss = Reward * actor_term - self.lamda * sparcity_term

    # custom actor loss
    custom_actor_loss = pt.mean(-custom_actor_loss)
    return combined_loss + custom_actor_loss

  def kl_reg(self, selection):
    alpha_0 = pt.tensor(self.alpha_0, dtype=pt.float32)
    hidden_dim = pt.tensor(selection.shape[1], dtype=pt.float32)

    trm1 = pt.lgamma(pt.sum(selection)) - pt.lgamma(hidden_dim * alpha_0)
    trm2 = - pt.sum(pt.lgamma(selection)) + hidden_dim * pt.lgamma(alpha_0)
    trm3 = pt.sum((selection - alpha_0) * (pt.digamma(selection) - pt.digamma(pt.sum(selection))))
    kl_d = trm1 + trm2 + trm3
    return self.kl_weight * kl_d / self.n_data

  def switch_loss(self, selection, log_critic_out, y_batch_scalar):
    critic_loss = nnf.nll_loss(log_critic_out, y_batch_scalar)
    return critic_loss + self.kl_reg(selection)

  def mse_match_loss(self, actor_out, log_critic_out, log_baseline_out):
    l_match = nnf.mse_loss(log_critic_out, log_baseline_out)
    return l_match + self.kl_reg(actor_out)

  def ce_match_loss(self, actor_out, log_critic_out, log_baseline_out):
    l_match = nnf.nll_loss(log_critic_out, pt.max(log_baseline_out, dim=1)[1].to(pt.long))
    return l_match + self.kl_reg(actor_out)

  def ce_loss(self, actor_out, log_critic_out, y_true):
    return nnf.nll_loss(log_critic_out, y_true) + self.kl_reg(actor_out)


  def switch_selection(self, x_in):
    # Generate a batch of selection probability
    phi = self.actor_net(x_in)
    # phi = nnf.softplus(actor_out)  # now the size of phi is mini_batch by input_dim
    selection = phi / pt.sum(phi, dim=1).unsqueeze(dim=1)
    return selection

  def run_training(self, x_train, y_train):
    """Train INVASE.

    Args:
      - x_train: training features
      - y_train: training labels
    """
    self.train(True)
    if self.model_type in {'mse_match', 'ce_match'}:
      self.pretrain_baseline(x_train, y_train)

    optimizer = pt.optim.Adam(params=self.total_parameters, lr=self.learning_rate, weight_decay=1e-3)
    x_train = x_train.astype(np.float32)
    # y_train = y_train.astype(np.int)

    for iter_idx in range(self.iteration):
      optimizer.zero_grad()
      # # Train critic
      # Select a random batch of samples
      idx = np.random.randint(0, x_train.shape[0], self.batch_size)
      x_batch = pt.tensor(x_train[idx, :], device=self.device)
      y_batch_onehot = pt.tensor(y_train[idx, :], device=self.device)
      y_batch_scalar = pt.max(y_batch_onehot, dim=1)[1]

      actor_out = self.switch_selection(x_batch)
      if self.model_type in {'mse_match', 'ce_match', 'ce'}:
        selection = actor_out
      else:
        selection = actor_out.detach()

      log_critic_out = self.critic_net(x_batch * selection)

      if self.model_type in {'invase', 'mse_match', 'ce_match'}:
        log_baseline_out = self.baseline_net(x_batch)
      elif self.model_type in {'invase_minus', 'ce'}:
        log_baseline_out = None
      else:
        raise ValueError

      # Train the actor
      # actor_out = self.actor_net(x_batch)
      if self.model_type in {'invase', 'invase_minus'}:
        full_loss = self.invase_loss(selection, log_critic_out, log_baseline_out,
                                     y_batch_onehot, y_batch_scalar, actor_out)
      elif self.model_type == 'mse_match':
        full_loss = self.mse_match_loss(actor_out, log_critic_out, log_baseline_out)
      elif self.model_type == 'ce_match':
        full_loss = self.ce_match_loss(actor_out, log_critic_out, log_baseline_out)
      elif self.model_type == 'ce':
        full_loss = self.ce_loss(actor_out, log_critic_out, y_batch_scalar)
      else:
        raise ValueError

      full_loss.backward()
      optimizer.step()

      if iter_idx % 100 == 0:
        matches = pt.max(log_critic_out, dim=1)[1] == y_batch_scalar
        # print(matches)
        critic_acc = pt.sum(matches.to(pt.float32)) / y_batch_scalar.shape[0]
        print(f'Iterations: {iter_idx}, critic acc: {np.round(critic_acc.item(), 4)}, '
              f'full loss: {np.round(full_loss.item(), 4)}')

  def pretrain_baseline(self, x_train, y_train, pretrain_steps=4000):
    self.train(True)
    x_train = x_train.astype(np.float32)
    optimizer = pt.optim.Adam(params=self.baseline_net.parameters(), lr=self.learning_rate, weight_decay=1e-3)

    for iter_idx in range(pretrain_steps):
      optimizer.zero_grad()
      idx = np.random.randint(0, x_train.shape[0], self.batch_size)
      x_batch = pt.tensor(x_train[idx, :], device=self.device)
      y_batch_onehot = pt.tensor(y_train[idx, :], device=self.device)
      y_batch_scalar = pt.max(y_batch_onehot, dim=1)[1]

      log_baseline_out = self.baseline_net(x_batch)
      baseline_loss = nnf.nll_loss(log_baseline_out, y_batch_scalar)

      baseline_loss.backward()
      optimizer.step()

      if iter_idx % 100 == 0:
        matches = pt.max(log_baseline_out, dim=1)[1] == y_batch_scalar
        baseline_acc = pt.sum(matches.to(pt.float32)) / y_batch_scalar.shape[0]
        print(f'Iterations: {iter_idx}, baseline acc: {np.round(baseline_acc.item(), 4)}, '
              f'baseline loss: {np.round(baseline_loss.item(), 4)}')

  def importance_score(self, x):
    """Return featuer importance score.
    Args:
      - x: feature
    Returns:
      - feature_importance: instance-wise feature importance for x
    """
    return_numpy = False
    if not isinstance(x, pt.Tensor):
      x = pt.tensor(x, device=self.device, dtype=pt.float32)
      return_numpy = True

    feature_importance = self.switch_selection(x)
    if return_numpy:
      feature_importance = feature_importance.cpu().detach().numpy()
    return feature_importance

  def predict(self, x):
    """Predict outcomes.

    Args:
      - x: feature

    Returns:
      - y_hat: predictions
    """
    return_numpy = False
    if not isinstance(x, pt.Tensor):
      x = pt.tensor(x, device=self.device, dtype=pt.float32)
      return_numpy = True
    # Generate a batch of selection probability
    selection = self.switch_selection(x)
    # Sampling the features based on the selection_probability
    # selection = pt.bernoulli(selection_probability)
    # Prediction
    # print(selection[:10])
    y_hat = pt.exp(self.critic_net(x * selection))
    if return_numpy:
      y_hat = y_hat.cpu().detach().numpy()
    return y_hat



def feature_performance_metric(ground_truth, importance_score):
  """
  since we can't rely on important features having a minimum weight, we take n as the number relevant ground
  truth features and then check how many of the top n activated features in the sample are relevant
  """
  n = importance_score.shape[0]

  tpr = np.zeros([n, ])
  fdr = np.zeros([n, ])

  # print(ground_truth[:3])
  print(importance_score[:3])
  n_rel_features = np.sum(ground_truth, axis=1).astype(np.int)
  # print(n_rel_features[:3])
  sorted_scores = np.sort(importance_score)
  # print(sorted_scores[:3])
  top_act = np.asarray([sorted_scores[i, -k] for i, k in zip(range(n), n_rel_features)])
  # print(top_act[:3])
  new_imp_score = importance_score - top_act[:, None]
  new_imp_score[new_imp_score >= 0] = 1.
  new_imp_score[new_imp_score < 0] = 0.
  # print(n, np.sum(importance_score), np.sum(new_imp_score), np.sum(ground_truth),
  #       np.sum(np.sum(new_imp_score, axis=1) == np.sum(ground_truth, axis=1)))

  importance_score = new_imp_score

  # For each sample
  for i in range(n):
    # tpr
    tpr_nom = np.sum(importance_score[i, :] * ground_truth[i, :])
    tpr_den = np.sum(ground_truth[i, :])
    tpr[i] = 100 * float(tpr_nom) / float(tpr_den + 1e-8)

    # fdr
    fdr_nom = np.sum(importance_score[i, :] * (1 - ground_truth[i, :]))
    fdr_den = np.sum(importance_score[i, :])
    fdr[i] = 100 * float(fdr_nom) / float(fdr_den + 1e-8)

  mean_tpr = np.mean(tpr)
  std_tpr = np.std(tpr)
  mean_fdr = np.mean(fdr)
  std_fdr = np.std(fdr)

  return mean_tpr, std_tpr, mean_fdr, std_fdr


def prediction_performance_metric(y_test, y_hat):
  """Performance metrics for prediction (AUC, APR, Accuracy).

  Args:
    - y_test: testing set labels
    - y_hat: prediction on testing set

  Returns:
    - auc: area under roc curve
    - apr: average precision score
    - acc: accuracy
  """
  # print(y_hat[:5])
  auc = roc_auc_score(y_test[:, 1], y_hat[:, 1])
  apr = average_precision_score(y_test[:, 1], y_hat[:, 1])
  acc = accuracy_score(y_test[:, 1], 1. * (y_hat[:, 1] > 0.5))

  return auc, apr, acc


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_type', choices=['syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6'], default='syn6', type=str)
  parser.add_argument('--train_no', help='the number of training data', default=10000, type=int)
  parser.add_argument('--test_no', help='the number of testing data', default=10000, type=int)
  parser.add_argument('--dim', help='the number of features', choices=[11, 100], default=11, type=int)
  parser.add_argument('--lamda', help='inavse hyper-parameter lambda', default=0.1, type=float)
  parser.add_argument('--actor_h_dim', help='hidden state dimensions for actor', default=100, type=int)
  parser.add_argument('--critic_h_dim', help='hidden state dimensions for critic', default=200, type=int)
  parser.add_argument('--n_layer', help='the number of layers', default=3, type=int)
  parser.add_argument('--batch_size', help='the number of samples in mini batch', default=1000, type=int)
  parser.add_argument('--iteration', help='the number of iteration', default=10000, type=int)
  parser.add_argument('--activation', help='activation function of the networks',
                      choices=['selu', 'relu'], default='relu', type=str)
  parser.add_argument('--learning_rate', help='learning rate of model training', default=1e-4, type=float)
  parser.add_argument('--model_type', help='inavse or invase- (without baseline)',
                      choices=['invase', 'invase_minus', 'mse_match', 'ce_match', 'ce'], default='ce', type=str)

  parser.add_argument('--alpha_0', type=float, default=0.001)
  parser.add_argument('--kl-weight', type=float, default=.1)

  parser.add_argument('--no-cuda', action='store_true', default=False)
  args = parser.parse_args()

  print('#################### generating data')
  # Generate dataset
  x_train, y_train, g_train = generate_dataset(n=args.train_no, dim=args.dim, data_type=args.data_type, seed=0)

  x_test, y_test, g_test = generate_dataset(n=args.test_no, dim=args.dim, data_type=args.data_type, seed=0)

  model_parameters = {'lamda': args.lamda,
                      'actor_h_dim': args.actor_h_dim,
                      'critic_h_dim': args.critic_h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration,
                      'activation': args.activation,
                      'learning_rate': args.learning_rate,
                      'alpha_0': args.alpha_0,
                      'n_data': args.train_no,
                      'kl_weight': args.kl_weight}

  device = pt.device("cuda" if not args.no_cuda else "cpu")

  print('#################### training model')
  # Train the model
  model = InvaseSwitch(x_train, y_train, args.model_type, model_parameters, device)

  model.run_training(x_train, y_train)
  model.train(False)

  print('#################### evaluating')
  # # Evaluation
  # Compute importance score
  g_hat = model.importance_score(x_test)
  # importance_score = 1. * (g_hat > 0.5)

  # Evaluate the performance of feature importance
  mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(g_test, g_hat)

  # Print the performance of feature importance
  print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '%, ' + 'TPR std: ' + str(np.round(std_tpr, 1)) + '%, ')
  print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '%, ' + 'FDR std: ' + str(np.round(std_fdr, 1)) + '%, ')

  # Predict labels
  y_hat = model.predict(x_test)

  # Evaluate the performance of feature importance
  auc, apr, acc = prediction_performance_metric(y_test, y_hat)

  # Print the performance of feature importance
  print('AUC: ' + str(np.round(auc, 3)) + ', APR: ' + str(np.round(apr, 3)) + ', ACC: ' + str(np.round(acc, 3)))

  performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr, 'mean_fdr': mean_fdr,
                 'std_fdr': std_fdr, 'auc': auc, 'apr': apr, 'acc': acc}

  return performance


##
if __name__ == '__main__':
  main()
