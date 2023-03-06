"""
This script is written for computing the entropy of switch distribution learned on Adult data
"""

__author__ = 'anon_m'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.stats import dirichlet

iter_sigmas = np.array([0., 1., 10., 50., 100.])
N = 20
entropy = np.zeros((N,len(iter_sigmas)))

dataset = 'adult'

for k in range(iter_sigmas.shape[0]):

    filename_phi = 'weights/%s_switch_parameter' % dataset + str(int(iter_sigmas[k]))
    switch_parameter_mat = np.load(filename_phi + '.npy')


    print('sigma is', str(int(iter_sigmas[k])))

    for i in range(N):
        entropy[i,k] = dirichlet.entropy(switch_parameter_mat[i, :])


x = iter_sigmas
y = np.mean(entropy,0)
dy = np.std(entropy,0)
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()
# plt.xscale('log')

