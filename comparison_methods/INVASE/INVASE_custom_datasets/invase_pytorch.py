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


class Net(nn.Module):
  def __init__(self, d_in, d_hid, d_out, act='relu', act_out='logsoftmax', use_bn=True):
    assert act in {'relu', 'selu'}
    assert act_out in {'logsoftmax', 'sigmoid'}
    super(Net, self).__init__()
    self.use_bn = use_bn
    self.act = nn.ReLU() if act == 'relu' else nn.SELU()
    self.act_out = nn.LogSoftmax(dim=1) if act_out == 'logsoftmax' else nn.Sigmoid()
    self.fc1 = nn.Linear(d_in, d_hid)
    self.bn1 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.bn2 = nn.BatchNorm1d(d_hid) if use_bn else None
    self.fc3 = nn.Linear(d_hid, d_out)

  def forward(self, x_in):
    x = self.act(self.fc1(x_in))
    x = self.bn1(x) if self.use_bn else x
    x = self.act(self.fc2(x))
    x = self.bn2(x) if self.use_bn else x
    x = self.act_out(self.fc3(x))
    return x


class Invase(nn.Module):
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
    super(Invase, self).__init__()

    self.lamda = model_parameters['lamda']
    self.actor_h_dim = model_parameters['actor_h_dim']
    self.critic_h_dim = model_parameters['critic_h_dim']
    self.n_layer = model_parameters['n_layer']
    self.batch_size = model_parameters['batch_size']
    self.iteration = model_parameters['iteration']
    self.activation = model_parameters['activation']
    self.learning_rate = model_parameters['learning_rate']

    self.device = device
    self.dim = x_train.shape[1]
    self.label_dim = 2# y_train.shape[1]

    self.model_type = model_type

    # Build and compile critic
    self.critic_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                          act_out='logsoftmax').to(self.device)
    # self.critic.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # Build and compile the actor
    self.actor_net = Net(self.dim, self.actor_h_dim, self.dim, act=self.activation,
                         act_out='sigmoid', use_bn=False).to(self.device)
    # self.actor.compile(loss=self.actor_loss, optimizer=optimizer)

    total_parameters = list(self.critic_net.parameters()) + list(self.actor_net.parameters())

    if self.model_type == 'invase':
      # Build and compile the baseline
      self.baseline_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
                              act_out='logsoftmax').to(self.device)
      # self.baseline.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
      total_parameters.extend(list(self.baseline_net.parameters()))

    self.optimizer = pt.optim.Adam(params=total_parameters, lr=self.learning_rate, weight_decay=1e-3)

  def actor_loss(self, selection, log_critic_out, log_baseline_out, y_true, actor_out):
    """Custom loss for the actor.

    Args:
      - y_true:
        - actor_out: actor output after sampling
        - critic_out: critic output
        - baseline_out: baseline output (only for invase)
      - y_pred: output of the actor network

    Returns:
      - loss: actor loss
    """

    # Critic loss
    # critic_loss = -pt.sum(y_true * pt.log(critic_out + 1e-8), dim=1)
    selection = selection.detach()
    critic_loss = -pt.sum(y_true * log_critic_out.detach(), dim=1)

    if self.model_type == 'invase':
      # Baseline loss
      baseline_loss = -pt.sum(y_true * log_baseline_out.detach(), dim=1)
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

    return custom_actor_loss

  def run_training(self, x_train, y_train):
    """Train INVASE.

    Args:
      - x_train: training features
      - y_train: training labels
    """
    self.train(True)

    x_train = x_train.astype(np.float32)
    # y_train = y_train.astype(np.int)

    for iter_idx in range(self.iteration):
      self.optimizer.zero_grad()
      # # Train critic
      # Select a random batch of samples
      idx = np.random.randint(0, x_train.shape[0], self.batch_size)
      x_batch = pt.tensor(x_train[idx, :], device=self.device)


      # four lines for custom datasets
      y_batch_scalar = pt.tensor(y_train[idx], device=self.device).long()
      y_batch_onehot = pt.zeros(y_batch_scalar.shape[0], 2, device=self.device)
      y_batch_onehot[y_batch_scalar == 0, 0] = 1
      y_batch_onehot[y_batch_scalar == 1, 1] = 1



      #2 lines for synthetic
      #y_batch_onehot = pt.tensor(y_train[idx, :], device=self.device)
      #y_batch_scalar = pt.max(y_batch_onehot, dim=1)[1]

      # print(y_batch_scalar)
      # print(y_batch_onehot[0], y_batch_scalar[0])

      # Generate a batch of selection probability
      actor_out = self.actor_net(x_batch)
      # Sampling the features based on the selection_probability
      selection = pt.bernoulli(actor_out)
      # Critic output
      log_critic_out = self.critic_net(x_batch * selection)
      # Critic loss
      critic_loss = nnf.nll_loss(log_critic_out, y_batch_scalar)

      combined_loss = critic_loss
      # # Train actor
      # Use multiple things as the y_true:
      # - selection, critic_out, baseline_out, and ground truth (y_batch)
      if self.model_type == 'invase':
        log_baseline_out = self.baseline_net(x_batch)
        baseline_loss = nnf.nll_loss(log_baseline_out, y_batch_scalar)
        combined_loss += baseline_loss
      elif self.model_type == 'invase_minus':
        log_baseline_out = None
      else:
        raise ValueError

      # Train the actor
      # actor_out = self.actor_net(x_batch)
      actor_loss = self.actor_loss(selection, log_critic_out, log_baseline_out,
                                   y_batch_onehot, actor_out)
      combined_loss += actor_loss

      combined_loss.backward()
      self.optimizer.step()

      if iter_idx % 100 == 0:
        matches = pt.max(log_critic_out, dim=1)[1] == y_batch_scalar
        # print(matches)
        critic_acc = pt.sum(matches.to(pt.float32)) / y_batch_scalar.shape[0]
        print(f'Iterations: {iter_idx}, critic acc: {np.round(critic_acc.item(), 4)}, '
              f'actor loss: {np.round(actor_loss.item(), 4)}')

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

    feature_importance = self.actor_net(x)
    if return_numpy:
      feature_importance = feature_importance.cpu().detach().numpy()
    return feature_importance

  def predict(self, x):
    """
    Predict outcomes.
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
    selection_probability = self.actor_net(x)
    # Sampling the features based on the selection_probability
    selection = pt.bernoulli(selection_probability)
    # Prediction
    y_hat = pt.exp(self.critic_net(x * selection))
    if return_numpy:
      y_hat = y_hat.cpu().detach().numpy()
    return y_hat
