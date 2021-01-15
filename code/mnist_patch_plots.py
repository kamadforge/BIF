import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt


def patch_plots_v1():
  base_dir = 'plots/patch_plots/'

  def aggregate_plot_mats(mats, save_file):
    print([k.shape for k in mats])
    agg_mat = np.concatenate([mats[0]] + [mats[k][28:, :] for k in range(1, len(mats))])

    np.save(base_dir + save_file + '.npy', agg_mat)
    plt.imsave(base_dir + save_file + '.png', agg_mat, vmin=0., vmax=1.)
  # switch_all_ks:
  plot_mats = [np.load(base_dir + f'switch_labels_38_k{k}_seed1.npy') for k in range(1, 6)]
  aggregate_plot_mats(plot_mats, '_switches_all_k')

  # l2x_all_ks:
  plot_mats = [np.load(base_dir + f'l2x_labels_38_k{k}_seed1.npy') for k in range(1, 6)]
  aggregate_plot_mats(plot_mats, '_l2x_all_k')

  # invase_all_ks:
  key_tuples = [(100.0, 5, 1.00), (50.0, 2, 2.00), (23.0, 1, 3.00), (18.5, 1, 4.00), (15.5, 2, 5.00)]  # lamda, seed, k
  plot_mats = [np.load(base_dir + f'invase_labels_38_lambda{k[0]}_seed{k[1]}_k_avg{k[2]:.2f}.npy') for k in key_tuples]
  aggregate_plot_mats(plot_mats, '_invase_all_k')


def patch_plots_v2():
  base_dir = 'plots/patch_plots/'
  methods = ['switches', 'l2x', 'invase']
  subdir = 'overlay_rows/'
  save_dir = base_dir + subdir
  os.makedirs(save_dir, exist_ok=True)
  for m in methods:
    mat = np.load(base_dir + f'_{m}_all_k.npy')
    # for each row (i.e. k), create a version with and a version without the the mnist number in the background
    digits = mat[:28]
    for k in range(1, 6):
      selection = mat[k*28:(k+1)*28, :, 2]
      x_sel = mat[k * 28:(k + 1) * 28, :, 0]
      overlay = np.minimum(digits, np.stack([1 - selection] * 3, axis=2))

      mat_no_overlay = np.stack([x_sel, np.zeros_like(x_sel), selection - x_sel], axis=2)
      mat_overlay = overlay + mat_no_overlay

      plt.imsave(save_dir + f'{m}_k{k}_only_selection.png', mat_no_overlay, vmin=0., vmax=1.)
      plt.imsave(save_dir + f'{m}_k{k}_with_digit.png', mat_overlay, vmin=0., vmax=1.)

      np.save(save_dir + f'{m}_k{k}_only_selection.npy', mat_no_overlay)
      np.save(save_dir + f'{m}_k{k}_with_digit.npy', mat_overlay)


if __name__ == '__main__':
  # patch_plots_v1()
  patch_plots_v2()