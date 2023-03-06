import numpy as np
import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
from Models import Feature_Importance_Model
from Losses import loss_function
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import matplotlib.pyplot as plt

# Main path: featimp_dp


class ImportedClassifier(nn.Module):
  def __init__(self, d_in, weights_file):
    super(ImportedClassifier, self).__init__()

    self.fc1 = nn.Linear(d_in, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, 32)
    self.fc4 = nn.Linear(32, 1)
    self.do1 = nn.Dropout(0.2)
    self.do2 = nn.Dropout(0.2)
    self.do3 = nn.Dropout(0.2)

    self.load_weights(weights_file)
    self.eval()

  def forward(self, x):
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.do1(x)
    x = self.fc2(x)
    x = nnf.relu(x)
    x = self.do2(x)
    x = self.fc3(x)
    x = nnf.relu(x)
    x = self.do3(x)
    x = self.fc4(x)
    x = pt.sigmoid(x)
    return x

  def load_weights(self, weights_file):
    w_dict = np.load(weights_file)
    load_dict = self.state_dict().copy()
    for idx in range(4):
      load_dict[f'fc{idx + 1}.weight'] = pt.tensor(w_dict[f'weight{idx}'].T)
      load_dict[f'fc{idx + 1}.bias'] = pt.tensor(w_dict[f'bias{idx}'])
    self.load_state_dict(load_dict)


def main():

  """ load data """
  X = np.load('X_adult_for_fairness.npy')
  y = np.load('y_adult_for_fairness.npy')

  N, input_dim = X.shape

  baseline = False  # for baseline classifier
  if baseline:
    classifier = ImportedClassifier(d_in=input_dim, weights_file='baseline_clf.npz')
  else:
    T_iter = 250 # either 1, 60, 125, 185, or 250
    filename = 'fair_clf_' + str(T_iter) + '.npz'
    classifier = ImportedClassifier(d_in=input_dim, weights_file=filename)


  maxseed = 5
  mean_importance = np.zeros((maxseed, input_dim))
  phi_store = np.zeros((maxseed, input_dim))

  for seednum in range(0,maxseed):
    np.random.seed(seednum)

    """ learn feature importance """
    num_Dir_samps = 1
    importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps)
    optimizer = optim.Adam(importance.parameters(), lr=0.075)

    # We freeze the classifier
    ct = 0
    for child in importance.children():
      ct += 1
      if ct >= 1:
        for param in child.parameters():
          param.requires_grad = False

    # print(list(importance.parameters())) # make sure I only update the gradients of feature importance

    importance.train()
    epoch = 400
    alpha_0 = 0.1
    annealing_rate = 1  # we don't anneal.
    kl_term = True

    for ind in range(epoch):
      optimizer.zero_grad()
      y_pred, phi_cand = importance(torch.Tensor(X))
      labels = torch.squeeze(torch.Tensor(y))
      loss = loss_function(y_pred, labels.view(-1, 1).repeat(1, num_Dir_samps), phi_cand, alpha_0, input_dim,
                           annealing_rate, N, kl_term)
      loss.backward()
      optimizer.step()

      # print(phi_cand/torch.sum(phi_cand))

      print('Epoch {}: training loss: {}'.format(ind, loss))

    """ checking the results """
    estimated_params = list(importance.parameters())
    phi_est = F.softplus(torch.Tensor(estimated_params[0]))
    concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
    beta_param = torch.ones(concentration_param.size())
    Gamma_obj = Gamma(concentration_param, beta_param)
    gamma_samps = Gamma_obj.rsample()
    Sstack = gamma_samps / torch.sum(gamma_samps, 0)
    avg_S = torch.mean(Sstack, 1)
    var_S = torch.var(Sstack,1)
    posterior_mean_switch = avg_S.detach().numpy()
    print('estimated posterior mean of feature importance is', posterior_mean_switch)

    phi_est = phi_est.detach().numpy()
    mean_importance[seednum,:] = posterior_mean_switch
    phi_store[seednum,:] = phi_est

  if baseline:
    filename = 'baseline_importance.npy'
    np.save(filename, mean_importance)
    filename = 'baseline_phi_est.npy'
    np.save(filename, phi_store)
  else:
    filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
    np.save(filename, mean_importance)
    filename = 'fair_clf_' + str(T_iter) + 'phi_est.npy'
    np.save(filename, phi_store)


  # Never married people
  # uval  = np.unique(X[:, 5])
  # people_who_never_married = X[:, 5] == uval[2]
  #
  # index_of_interest = np.where(1*people_who_never_married==1)
  # np.sum(y[people_who_never_married])/N
  # np.sum(y)/N

if __name__ == '__main__':
  main()
