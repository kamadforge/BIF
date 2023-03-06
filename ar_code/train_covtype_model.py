import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import os


class CovTypeDataset(Dataset):
  def __init__(self, train, test_split=0.2, deterministic=True, normalize=True, force_make_data=False):
    super(CovTypeDataset, self).__init__()
    self.train = train

    spc = f'{test_split}_{"det" if deterministic else ""}_{"normed" if normalize else ""}{"train" if train else "test"}'
    load_str = f'../data/covtype/split{spc}.npz'

    if not os.path.exists(load_str) or force_make_data:
      self.make_balanced_split(test_split, deterministic, load_str, normalize)
    else:
      print('loading existing dataset:', load_str)

    mat = np.load(load_str)
    self.inputs = mat['x']
    self.labels = mat['y']

  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return pt.tensor(self.inputs[idx]), pt.tensor(self.labels[idx])

  @staticmethod
  def make_balanced_split(test_split, deterministic, save_str, normalize):
    original_train_data = np.load("../data/covtype/train.npy")
    original_test_data = np.load("../data/covtype/test.npy")

    # we put them together and make a new train/test split in the following
    original_data = np.concatenate((original_train_data, original_test_data))

    labels = original_data[:, -1]
    inputs = original_data[:, :-1]

    if normalize:
      feature_means = np.mean(inputs, axis=0)
      feature_sdevs = np.std(inputs, axis=0)
      inputs = (inputs - feature_means) / feature_sdevs
      print('before normalizing')
      print(f'means {feature_means}')
      print(f'sdevs {feature_sdevs}')
      print('after normalizing')
      print(f'means {np.mean(inputs, axis=0)}')
      print(f'sdevs {np.std(inputs, axis=0)}')

    n_data = inputs.shape[0]  # 581012
    n_labels = int(np.max(labels) + 1)
    n_test_per_label = int(np.floor(n_data * test_split / n_labels))
    n_train_per_label = n_data // n_labels - n_test_per_label

    n_data_per_label = n_train_per_label if train else n_test_per_label
    inputs_by_label = []

    def balance_label_data(label_data, target_n_samples):
      n_label_data = label_data.shape[0]
      n_samples = int(target_n_samples % n_label_data)
      n_repeats = int(target_n_samples // n_label_data)

      print(f'nsamples {n_samples}, nrepeats {n_repeats}')

      if deterministic:
        sample_data = label_data[:n_samples]
      else:
        rand_perm = np.random.permutation(n_label_data)
        sample_data = label_data[rand_perm][:n_samples]

      if n_repeats > 0:
        repeat_data = np.concatenate([label_data] * n_repeats)
        return np.concatenate([repeat_data, sample_data])
      else:
        return sample_data

    print(f'n_data={n_data}')
    for label in range(n_labels):
      inputs_i = inputs[labels == label]
      n_data_i = inputs_i.shape[0]

      print(f'label {label} has {n_data_i} instances ({n_data_i / labels.shape[0]}%)')

      # we now fully balance the data by dividing data for each class into train and test split, and then sampling equally
      n_test_i = int(np.floor(n_data_i * test_split))
      n_train_i = n_data_i - n_test_i

      selected_inputs_i = inputs_i[:n_train_i] if train else inputs_i[n_train_i:]
      inputs_by_label.append(balance_label_data(selected_inputs_i, n_data_per_label))

      print(f'ntrain: {n_train_i} ntest: {n_test_i}')
      print(f'balanced data {inputs_by_label[-1].shape[0]}')

      balanced_inputs = np.concatenate(inputs_by_label)
      balanced_labels = np.concatenate([np.ones((n_data_per_label,)) * k for k in range(n_labels)])
      train_perm = np.random.permutation(balanced_inputs.shape[0])
      balanced_inputs = balanced_inputs[train_perm]
      balanced_labels = balanced_labels[train_perm].astype(np.int)

      np.savez(save_str, x=balanced_inputs, y=balanced_labels)


def get_covtype_dataloaders(use_cuda, batch_size, test_batch_size):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_data = CovTypeDataset(train=True)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
  test_data = CovTypeDataset(train=False)
  test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


class CovtypeNet(nn.Module):
  def __init__(self, d_in=54, selected_label=None):
    super(CovtypeNet, self).__init__()
    self.selected_label = selected_label
    self.fc1 = nn.Linear(d_in, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = pt.flatten(x, 1)
    x = self.fc1(x)
    x = nnf.relu(x)
    x = self.fc2(x)
    if self.selected_label is None:
      output = nnf.log_softmax(x, dim=1)
    else:
      output = nnf.softmax(x, dim=1)[:, self.selected_label]
    return output


def train(args, model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = nnf.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with pt.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += nnf.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=100)
  parser.add_argument('--epochs', type=int, default=7)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--gamma', type=float, default=0.7)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=500)
  parser.add_argument('--reduce-features', action='store_true', default=True)
  parser.add_argument('--save-model', action='store_true', default=True)

  args = parser.parse_args()
  use_cuda = not args.no_cuda and pt.cuda.is_available()

  pt.manual_seed(args.seed)

  device = pt.device("cuda" if use_cuda else "cpu")

  train_loader, test_loader = get_covtype_dataloaders(use_cuda, args.batch_size, args.test_batch_size)

  model = CovtypeNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

  if args.save_model:
    pt.save(model.state_dict(), f"models/covtype_nn_ep{args.epochs}{'_10feats' if args.reduce_features else ''}.pt")


if __name__ == '__main__':
  main()


# OLD INIT
# super(CovTypeDataset, self).__init__()
#     self.train = train
#
#     train_data = np.load("../data/covtype/train.npy")
#     test_data = np.load("../data/covtype/test.npy")
#
#     # we put them together and make a new train/test split in the following
#     data = np.concatenate((train_data, test_data))
#
#     labels = data[:, -1]
#     data = data[:, :-1]
#
#     n_data = data.shape[0]  # 581012
#     n_labels = int(np.max(labels) + 1)
#     n_test_per_label = int(np.floor(n_data * test_split / n_labels))
#     n_train_per_label = n_data // n_labels - n_test_per_label
#
#     train_by_label = []
#     test_by_label = []
#
#     def balance_label_data(label_data, target_n_samples):
#       n_label_data = label_data.shape[0]
#       n_samples = int(target_n_samples % n_label_data)
#       n_repeats = int(target_n_samples // n_label_data)
#
#       print(f'nsamples {n_samples}, nrepeats {n_repeats}')
#
#       if deterministic:
#         sample_data = label_data[:n_samples]
#       else:
#         rand_perm = np.random.permutation(n_label_data)
#         sample_data = label_data[rand_perm][:n_samples]
#
#       if n_repeats > 0:
#         repeat_data = np.concatenate([label_data] * n_repeats)
#         return np.concatenate([repeat_data, sample_data])
#       else:
#         return sample_data
#
#     print(f'n_data={n_data}')
#     for label in range(n_labels):
#       data_i = data[labels == label]
#       n_data_i = data_i.shape[0]
#
#       print(f'label {label} has {n_data_i} instances ({n_data_i / labels.shape[0]}%)')
#
#       # we now fully balance the data by dividing data for each class into train and test split, and then sampling equally
#       n_test_i = int(np.floor(n_data_i * test_split))
#       n_train_i = n_data_i - n_test_i
#
#       test_by_label.append(balance_label_data(data_i[:n_test_i], n_test_per_label))
#       train_by_label.append(balance_label_data(data_i[n_test_i:], n_train_per_label))
#
#       print(f'ntrain: {n_train_i} ntest: {n_test_i}')
#       print(f'balanced ntrain {train_by_label[-1].shape[0]} ntest: {test_by_label[-1].shape[0]}')
#
#     if train:
#       train_data = np.concatenate(train_by_label)
#       train_labels = np.concatenate([np.ones((n_train_per_label,)) * k for k in range(n_labels)])
#       train_perm = np.random.permutation(train_data.shape[0])
#       self.data = train_data[train_perm]
#       self.labels = train_labels[train_perm]
#     else:
#       test_data = np.concatenate(test_by_label)
#       test_labels = np.concatenate([np.ones((n_test_per_label,)) * k for k in range(n_labels)])
#       test_perm = np.random.permutation(test_data.shape[0])
#       self.data = test_data[test_perm]
#       self.labels = test_labels[test_perm]