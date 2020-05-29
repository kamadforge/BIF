import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf

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
  base_clf = ImportedClassifier(d_in=94, weights_file='baseline_clf.npz')
  fair_clf = ImportedClassifier(d_in=94, weights_file='fair_clf.npz')
  # print(base_clf(pt.zeros(3, 94)))
  print(fair_clf(pt.zeros(1, 94)))
  print(fair_clf(pt.ones(1, 94)))
  # clf.load_state_dict()

if __name__ == '__main__':
  main()
