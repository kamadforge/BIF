import numpy as np
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
  labels = [0, 1, 2, 3]
  loadnames = [f'weights/covtype_switch_posterior_mean_sig0_label{k}.npy' for k in labels]
  switch_mats = np.concatenate([np.load(k) for k in loadnames])
  plt.imshow(switch_mats, cmap=cm.get_cmap('viridis'))
  plt.savefig('covtype_plot.png')


if __name__ == '__main__':
  main()
