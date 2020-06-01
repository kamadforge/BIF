"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'mijung'
import argparse
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
from torch.utils.data import Dataset, DataLoader

import pickle
from sklearn.model_selection import train_test_split



class ImportedDPClassifier(nn.Module):
  def __init__(self, d_in, weights_file):
    super(ImportedDPClassifier, self).__init__()
    self.input_size = 12
    self.hidden_size = 100
    self.hidden_size2 = 20

    self.fc1 = nn.Linear(d_in, 100)
    self.fc2 = nn.Linear(100, 20)
    self.fc3 = nn.Linear(20, 1)
    self.load_weights(weights_file)
    self.eval()

  def forward(self, x):
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.fc2(x)
    x = nnf.relu(x)
    x = self.fc3(x)
    x = pt.sigmoid(x)
    return x

  def load_weights(self, weights_file):
    loaded_states = pt.load(weights_file)
    clean_states = dict()
    for key in self.state_dict().keys():
      clean_states[key] = loaded_states[key]
    self.load_state_dict(clean_states)


class AdultDataset(Dataset):
  def __init__(self, train, data_file='code/adult.p'):
    super(AdultDataset, self).__init__()
    self.train = train
    self.data_file = data_file
    self.inputs, self.labels = self.load_dataset()

  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return pt.tensor(self.inputs[idx]), pt.tensor(self.labels[idx])

  def load_dataset(self):

    with open(self.data_file, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
      y_tot, x_tot = data

    x_tot = x_tot.astype(np.float32)
    y_tot = y_tot.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(x_tot, y_tot, test_size=0.5, stratify=y_tot, random_state=7)


    if self.train:
      return x_train, y_train
    else:
      return x_test, y_test


def get_adult_dataloaders(use_cuda, batch_size, test_batch_size, data_file):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_data = AdultDataset(train=True, data_file=data_file)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = AdultDataset(train=False, data_file=data_file)
  test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def invase_select_data(invase_model, loader, device):
  x_data, y_data, selection, selection_prob = [], [], [], []
  with pt.no_grad():
    for x, y in loader:
      x_data.append(x.numpy())
      y_data.append(y.numpy())
      # x_sel = nnf.softplus(selector(x.to(device)))
      # x_sel = x_sel / pt.sum(x_sel, dim=1)[:, None] * 16  # multiply by patch size
      x_sel, x_prob = invase_model.select(x.to(device))
      selection.append(x_sel.cpu().numpy())
      selection_prob.append(x_prob.cpu().numpy())

  return np.concatenate(x_data), np.concatenate(y_data), np.concatenate(selection), np.concatenate(selection_prob)


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
    assert not pt.any(pt.isnan(x_in)).item()
    x = self.fc1(x_in)
    assert not pt.any(pt.isnan(x)).item()
    x = self.bn1(x) if self.use_bn else x
    x = self.act(x)
    x = self.fc2(x)
    assert not pt.any(pt.isnan(x)).item()
    x = self.bn2(x) if self.use_bn else x
    x = self.act(x)
    x = self.fc3(x)
    assert not pt.any(pt.isnan(x)).item()
    x = self.act_out(x)
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
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE
  """

  def __init__(self, model_parameters, device, clf_weights_file):
    super(Invase, self).__init__()

    self.lamda = model_parameters['lamda']
    self.actor_h_dim = model_parameters['actor_h_dim']
    self.critic_h_dim = model_parameters['critic_h_dim']
    self.batch_size = model_parameters['batch_size']
    self.activation = model_parameters['activation']
    self.learning_rate = model_parameters['learning_rate']
    self.model_type = model_parameters['model_type']

    self.device = device
    self.dim = 14
    self.label_dim = 2


    # Build and compile critic
    # self.critic_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
    #                       act_out='logsoftmax').to(self.device)
    self.critic_net = ImportedDPClassifier(d_in=self.dim, weights_file=clf_weights_file).to(self.device)


    # Build and compile the actor
    self._actor_net = Net(self.dim, self.actor_h_dim, self.dim, act=self.activation,
                          act_out='sigmoid', use_bn=False).to(self.device)

    if self.model_type == 'invase':
      # Build and compile the baseline
      # self.baseline_net = Net(self.dim, self.critic_h_dim, self.label_dim, act=self.activation,
      #                         act_out='logsoftmax').to(self.device)
      self.baseline_net = ImportedDPClassifier(d_in=self.dim, weights_file=clf_weights_file).to(self.device)

    self.optimizer = pt.optim.Adam(params=self._actor_net.parameters(), lr=self.learning_rate, weight_decay=1e-3)

  def actor_loss(self, selection, critic_out, baseline_out, y_true, actor_out):
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
    assert not pt.any(pt.isnan(critic_out))
    log_critic_out = pt.log(critic_out.detach() + 1e-8)
    assert not pt.any(pt.isnan(log_critic_out))
    assert not pt.any(pt.isnan(y_true))
    selection = selection.detach()
    critic_loss = -pt.sum(y_true * log_critic_out, dim=1)
    assert not pt.any(pt.isnan(critic_loss))
    if self.model_type == 'invase':
      # Baseline loss
      baseline_loss = -pt.sum(y_true * pt.log(baseline_out.detach() + 1e-8), dim=1)
      # Reward
      Reward = -(critic_loss - baseline_loss)
    elif self.model_type == 'invase_minus':
      Reward = -critic_loss
    else:
      raise ValueError

    # Policy gradient loss computation.
    # actor_term = pt.sum(selection * pt.log(actor_out + 1e-8) + (1 - selection) * pt.log(1 - actor_out + 1e-8), dim=1)
    actor_term = pt.sum(selection * pt.log(actor_out + 1e-8) + (1 - selection) * pt.log(1 - actor_out + 1e-8), dim=1)
    sparcity_term = pt.mean(actor_out, dim=1)
    custom_actor_loss = Reward * actor_term - self.lamda * sparcity_term

    # custom actor loss
    custom_actor_loss = pt.mean(-custom_actor_loss)

    return custom_actor_loss

  def importance_score(self, x):
    """Return feature importance score.

    Args:
      - x: feature

    Returns:
      - feature_importance: instance-wise feature importance for x
    """
    return_numpy = False
    if not isinstance(x, pt.Tensor):
      x = pt.tensor(x, device=self.device, dtype=pt.float32)
      return_numpy = True

    _, feature_importance = self.select(x)
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
    selection, _ = self.select(x)
    # Sampling the features based on the selection_probability
    # selection = pt.bernoulli(selection_probability)
    # Prediction
    y_hat = pt.exp(self.critic_net(x * selection))
    if return_numpy:
      y_hat = y_hat.cpu().detach().numpy()
    return y_hat

  def select(self, x):
    actor_out = self._actor_net(x)
    assert not pt.any(pt.isnan(actor_out)).item()
    min_out, max_out = pt.min(actor_out), pt.max(actor_out)
    assert min_out >= 0. and max_out <= 1.
    selection = pt.bernoulli(actor_out)

    # upscale to patches, first output, then sampled selection
    # Sampling the features based on the selection_probability
    return selection, actor_out


def train_model(model, learning_rate, n_epochs, train_loader, test_loader, device):
  # adam = pt.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-3)

  # filepath = f"models/mnist/model.pt"

  for ep in range(n_epochs):

    model._actor_net.train()
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1).to(device)
      assert not pt.any(pt.isnan(x_batch)).item()
      assert not pt.any(pt.isnan(y_batch)).item()

      y_batch_onehot = pt.stack([1-y_batch, y_batch], dim=1)

      model.optimizer.zero_grad()
      selection, actor_out = model.select(x_batch)
      assert not pt.any(pt.isnan(selection)).item()
      assert not pt.any(pt.isnan(actor_out)).item()

      critic_out = model.critic_net(x_batch * selection)
      critic_out = pt.cat([1 - critic_out, critic_out], dim=1).detach()
      # critic_loss = nnf.nll_loss(log_critic_out, y_batch)
      # combined_loss = critic_loss
      assert pt.min(y_batch) >= 0. and pt.max(y_batch) <= 1.
      assert pt.min(critic_out) >= 0. and pt.max(critic_out) <= 1.

      if model.model_type == 'invase':
        baseline_out = model.baseline_net(x_batch)
        baseline_out = pt.cat([1 - baseline_out, baseline_out], dim=1).detach()
        # baseline_loss = nnf.nll_loss(log_baseline_out, y_batch)
        # combined_loss += baseline_loss
      elif model.model_type == 'invase_minus':
        baseline_out = None
      else:
        raise ValueError

      actor_loss = model.actor_loss(selection, critic_out, baseline_out, y_batch_onehot, actor_out)
      actor_loss.backward()

      assert not any([pt.any(pt.isnan(k.grad)).item() for k in model._actor_net.parameters()])


      model.optimizer.step()

    model._actor_net.eval()

    summed_loss = 0
    correct_preds = 0
    n_tested = 0
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      x_batch = x_batch.reshape(x_batch.shape[0], -1)

      test_selection, _ = model.select(x_batch)
      critic_pred = model.critic_net(x_batch * test_selection)
      critic_pred = pt.cat([1 - critic_pred, critic_pred], dim=1)
      loss = nnf.nll_loss(critic_pred, y_batch)

      summed_loss += loss.item() * y_batch.shape[0]
      matches = pt.max(critic_pred, dim=1)[1] == y_batch
      correct_preds += pt.sum(matches).item()
      n_tested += y_batch.shape[0]

    print(f'epoch {ep} done. Acc: {correct_preds / n_tested}, Loss: {summed_loss / n_tested}')


def test_classifier_epoch(classifier, test_loader, device):
  classifier.eval()
  test_loss = 0
  correct = 0
  with pt.no_grad():
    for x_batch, y_batch in test_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      pred = classifier(x_batch)
      assert pt.min(y_batch) >= 0. and pt.max(y_batch) <= 1.
      assert pt.min(pred) >= 0. and pt.max(pred) <= 1.
      test_loss = nnf.binary_cross_entropy_with_logits(pred, y_batch, reduction='sum').item()
      # test_loss = nnf.nll_loss(pred, y_batch, reduction='sum').item()  # sum up batch loss
      class_pred = pt.sigmoid(pred).round()  # get the index of the max log-probability
      # print(class_pred.shape, y_batch.shape)
      correct += class_pred.eq(y_batch.view_as(class_pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=2)
  # parser.add_argument('--dataset', type=str, default='mnist')
  # parser.add_argument('--selected-label', type=int, default=3)  # label for 1-v-rest training
  # parser.add_argument('--log-interval', type=int, default=500)
  # parser.add_argument('--n-switch-samples', type=int, default=3)

  parser.add_argument('--lamda', help='inavse hyper-parameter lambda', default=.5, type=float)
  parser.add_argument('--actor_h_dim', help='hidden state dimensions for actor', default=100, type=int)
  parser.add_argument('--critic_h_dim', help='hidden state dimensions for critic', default=200, type=int)
  parser.add_argument('--activation', help='activation function of the networks',
                      choices=['selu', 'relu'], default='relu', type=str)
  parser.add_argument('--learning_rate', help='learning rate of model training', default=1e-4, type=float)
  parser.add_argument('--model_type', help='inavse or invase- (without baseline)',
                      choices=['invase', 'invase_minus'], default='invase_minus', type=str)

  # parser.add_argument('--clf-weights-file', type=str, default='../../code/Trade_offs/fair_clf_250.npz')
  parser.add_argument('--epsilon', type=str, default=0.5)


  # parser.add_argument("--freeze-classifier", default=True)
  # parser.add_argument("--patch-selection", default=True)

  return parser.parse_args()


def aggregate_global_importance(dataset_selection_probs, save_file):
  mean_probs = np.mean(dataset_selection_probs, axis=0)
  print('saving probabilities to file:', save_file)
  print(mean_probs)
  # print('shape of saved mean probabilities', mean_probs.shape)
  np.save(save_file, mean_probs)


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")
  # train_loader, test_loader = load_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size)
  # train_loader, test_loader = load_two_label_mnist_data(use_cuda, ar.batch_size, ar.test_batch_size,
  #                                                       data_path='../../data',
  #                                                       label_a=ar.label_a, label_b=ar.label_b,
  #                                                       tgt_type=np.int64)
  train_loader, test_loader = get_adult_dataloaders(use_cuda, ar.batch_size, ar.test_batch_size,
                                                    data_file='../../code/adult.p')

  model_parameters = {'lamda': ar.lamda,
                      'actor_h_dim': ar.actor_h_dim,
                      'critic_h_dim': ar.critic_h_dim,
                      'batch_size': ar.batch_size,
                      'activation': ar.activation,
                      'learning_rate': ar.learning_rate,
                      'model_type': ar.model_type}
  eps_to_sig = {0.5: '17.0', 1.: '8.4', 2.: '4.4', 4.: '2.3', 8.: '1.35', None:'0.0'}


  clf_weights_file = f'../../code/Trade_offs/dp_classifier_sig{eps_to_sig[ar.epsilon]}.pt'

  model = Invase(model_parameters, device, clf_weights_file)
  # d_in=784, d_out=1, datatype=None, n_key_features=ar.select_k, device=device).to(device)
  train_model(model, ar.lr, ar.epochs, train_loader, test_loader, device)
  # , ar.point_estimate, ar.n_switch_samples, ar.alpha_0, n_features, n_data, ar.KL_reg)

  print('Finished Training Selector')
  x_ts, y_ts, ts_selection, ts_selection_prob = invase_select_data(model, test_loader, device)
  x_tr, y_tr, tr_selection, tr_selection_prob = invase_select_data(model, train_loader, device)


  aggregate_global_importance(tr_selection_prob, f'private_global_importance_trainset_eps{ar.epsilon}.npy')
  aggregate_global_importance(ts_selection_prob, f'private_global_importance_testset_{ar.epsilon}.npy')

  # y_ts = y_ts.astype(np.int64)
  # x_ts_select = x_ts * ts_selection
  print('average number of selected patches: ', np.mean(np.sum(ts_selection, axis=1)))
  # x_ts_select = hard_select_data(x_ts, ts_selection, k=ar.select_k)
  # select_test_loader = make_select_loader(x_ts_select, y_ts, train=False, batch_size=ar.test_batch_size,
  #                                         use_cuda=use_cuda, data_path='../../data')
  # print('testing classifier')

  # test_posthoc_acc(ar.label_a, ar.label_b, select_test_loader, device, model_path_prefix='../../code/')


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  do_featimp_exp(ar)


if __name__ == '__main__':
  main()


