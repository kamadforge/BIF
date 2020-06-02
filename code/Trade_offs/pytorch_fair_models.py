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

  np.random.seed(0)

  baseline = False # for baseline classifier

  """ load data """
  X = np.load('X_adult_for_fairness.npy')
  y = np.load('y_adult_for_fairness.npy')

  # X = np.load('X_adult_for_fairness_longer.npy')
  # y = np.load('y_adult_for_fairness_longer.npy')

  N, input_dim = X.shape

  if baseline:
    classifier = ImportedClassifier(d_in=input_dim, weights_file='baseline_clf.npz')
  else:
    T_iter = 1  # either 1, 125 or 250
    filename = 'fair_clf_' + str(T_iter) + '.npz'
    classifier = ImportedClassifier(d_in=input_dim, weights_file=filename)


  """ learn feature importance """
  num_Dir_samps = 10
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
  annealing_rate = 1  # we don't anneal. don't want to think about this.
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

  if baseline:
    filename = 'baseline_importance.npy'
    np.save(filename, posterior_mean_switch)
    filename = 'baseline_phi_est.npy'
    np.save(filename, phi_est)
  else:
    filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
    np.save(filename, posterior_mean_switch)
    filename = 'fair_clf_' + str(T_iter) + 'phi_est.npy'
    np.save(filename, phi_est)


  # Never married people
  uval  = np.unique(X[:, 5])
  people_who_never_married = X[:, 5] == uval[2]

  index_of_interest = np.where(1*people_who_never_married==1)

  # plt.hist(y[index_of_interest])
  # plt.show()

  np.sum(y[people_who_never_married])/N
  np.sum(y)/N


  # shorter data
  # data = [
  #     age(0), workclass(1), fnlwgt(2), education(3), education_num(4),
  #     marital_status(5), occupation(6), relationship(7),
  #     capital_gain(8), capital_loss(9), hours_per_week(10), native_country(11)]


  # feature importance under baseline classifier
  # [0.00281593 0.00142916 0.00141594 0.00126064 0.3443077  0.50392526
  #  0.00118415 0.00150288 0.13795751 0.00115861 0.0015941  0.00144811]

  # feature importance when T=1 (alpha_0 = 0.1)
  #  [0.00394763 0.00133567 0.00118472 0.00110636 0.2889137  0.5241561
  #  0.00114463 0.00180197 0.17161408 0.00161925 0.00167312 0.00150287]

 # with alpha_0 = 0.01 , not so much difference. I will continue with alpha_0 = 0.1
 #[2.9165123e-03 2.4319009e-04 3.7485830e-04 2.2752557e-04 2.9997146e-01 5.2147448e-01
  # 4.1112691e-04 4.0848891e-04 1.7258091e-01 4.2654391e-04 5.1496323e-04 4.4992895e-04]


  # feature importance when T = 125 (alpha_0=0.1)
  #  [0.05821808 0.0013884  0.00120657 0.00145666 0.20682077 0.568012
  # 0.00135323 0.00106091 0.15579899 0.00135324 0.00164829 0.00168276]


  # feature importance when T = 250
  # [0.1005158  0.00124442 0.00113049 0.00142828 0.20497444 0.5148057
  # 0.00125945 0.00088737 0.16951144 0.00167988 0.00141403 0.00114866]






  # print(base_clf(pt.zeros(3, 94)))
  # print(fair_clf(pt.zeros(1, 94)))
  # print(fair_clf(pt.ones(1, 94)))
  # clf.load_state_dict()


if __name__ == '__main__':
  main()
