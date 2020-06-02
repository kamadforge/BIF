from autodp import rdp_acct, rdp_bank


def main():
  """ input arguments """

  # sig = 1.35 -> eps 8.07 ~= 8
  # sig = 2.3 -> eps 4.01 ~= 4
  # sig = 4.4 -> eps 1.94 ~= 2
  # sig = 8.4 -> eps 0.984 ~= 1
  # sig = 17. -> eps 0.48 ~= 0.5
  sigma = 17.

  # (2) desired delta level
  delta = 1e-5

  # (5) number of training steps
  n_epochs = 20  # 5 for DP-MERF and 17 for DP-MERF+AE
  batch_size = 1000  # the same across experiments

  n_data = 29304  # fixed for mnist
  steps_per_epoch = n_data // batch_size
  n_steps = steps_per_epoch * n_epochs
  # n_steps = 1

  # (6) sampling rate
  prob = batch_size / n_data
  # prob = 1

  """ end of input arguments """

  """ now use autodp to calculate the cumulative privacy loss """
  # declare the moment accountants
  acct = rdp_acct.anaRDPacct()

  for i in range(1, n_steps+1):
    acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
    # if i % steps_per_epoch == 0 or i == n_steps:
    if i == n_steps:
      print("[", i, "]Privacy loss is", acct.get_eps(delta))


if __name__ == '__main__':
  main()
