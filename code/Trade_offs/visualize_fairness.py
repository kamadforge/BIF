"""
This script is written for visualizing the results of Adult data for fairness
"""

__author__ = 'mijung'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from distance_Dirichlet import L2dist, expected_suff_stats, KL_Dir, KL_Bern

""" first plot : accuracy vs fairness"""
# load metrics
# eval_metrics = np.load('eval_metrics.npy') # first column: ROC, second column: Accuracy
# fairness_metrics = np.load('fairness_metrics.npy') # first column: Race, second column: Sex

fairness_metrics = pd.read_pickle("fairness_metrics.pkl")
eval_metrics = pd.read_pickle("eval_metrics.pkl")

sns.set(style="whitegrid")
# plt.subplot(211)
plt.figure(1)
# plt.plot(fairness_metrics["race"], eval_metrics["ROC AUC"])
ax = sns.lineplot(fairness_metrics[0], eval_metrics["ROC AUC"], palette="tab10", linewidth=2.5)
ax.set(xlabel='Fairness in terms of Race', ylabel='ROC AUC')
plt.savefig('fairness_vs_roc.pdf')
# plt.subplot(212)
# sns.lineplot(fairness_metrics["sex"], eval_metrics["Accuracy"], palette="tab10", linewidth=2.5)


# """ second plot : learned features when T=1, 125 and 250 """
mean_importance_baseline = np.load('baseline_importance.npy')
mean_importance_baseline = np.mean(mean_importance_baseline,0)
# phi_estimate_baseline = np.load('baseline_phi_est.npy')

plt.figure(2)
# https://seaborn.pydata.org/examples/color_palettes.html
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,12)
sns.barplot(x=x, y=mean_importance_baseline, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("baseline")
ax1.set_ylim(0,0.55)
ax1.set_title('Q-FIT')

fairness_level = fairness_metrics[0].to_numpy()

T_iter = 1
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
mean_importance = np.mean(mean_importance,0)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter]
ax2.set_ylabel(str(np.int(level))+"% Fair")
ax2.set_ylim(0,0.55)

T_iter = 60
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
mean_importance = np.mean(mean_importance,0)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter-1]
ax3.set_ylabel(str(np.int(level))+"% Fair")
ax3.set_ylim(0,0.55)

T_iter = 125
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
mean_importance = np.mean(mean_importance,0)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter]
ax4.set_ylabel(str(np.int(level))+"% Fair")
ax4.set_ylim(0,0.55)

T_iter = 185
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
mean_importance = np.mean(mean_importance,0)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax5)
ax5.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter-1]
ax5.set_ylabel(str(np.int(level))+"% Fair")
ax5.set_ylim(0,0.55)


plt.savefig('learned_importance.pdf')


""" third plot : distance of feature importance vs fairness """
# run code to learn phi for baseline, T=60,125,185,250 multiple times
# then compute the distance for each seed, then show the errorbar in this plot.

phi_estimate_baseline = np.load('baseline_phi_est.npy')
phi_estimate_baseline = np.mean(phi_estimate_baseline,0)
print("baseline phi: ", phi_estimate_baseline)

T=1
filename = 'fair_clf_' + str(T) + 'phi_est.npy'
phi_estimate_1 = np.load(filename)
phi_estimate_1 = np.mean(phi_estimate_1,0)
print("phi at T=1: ", phi_estimate_1)

T = 60
filename = 'fair_clf_' + str(T) + 'phi_est.npy'
phi_estimate_60 = np.load(filename)
phi_estimate_60 = np.mean(phi_estimate_60,0)
print("phi at T=60: ", phi_estimate_60)

T = 125
filename = 'fair_clf_' + str(T) + 'phi_est.npy'
phi_estimate_125 = np.load(filename)
phi_estimate_125 = np.mean(phi_estimate_125,0)
print("phi at T=125: ", phi_estimate_125)

T = 185
filename = 'fair_clf_' + str(T) + 'phi_est.npy'
phi_estimate_185 = np.load(filename)
phi_estimate_185 = np.mean(phi_estimate_185,0)
print("phi at T=185: ", phi_estimate_185)


# T = 250
# filename = 'fair_clf_' + str(T) + 'phi_est.npy'
# phi_estimate_250 = np.load(filename)
# phi_estimate_250 = np.mean(phi_estimate_250,0)
# print("phi at T=250: ", phi_estimate_250)


# # L2distance in suff stats
# L2_1 = L2dist(expected_suff_stats(phi_estimate_baseline), expected_suff_stats(phi_estimate_1))
# L2_60 = L2dist(expected_suff_stats(phi_estimate_baseline), expected_suff_stats(phi_estimate_60))
# L2_125 = L2dist(expected_suff_stats(phi_estimate_baseline), expected_suff_stats(phi_estimate_125))
# L2_185 = L2dist(expected_suff_stats(phi_estimate_baseline), expected_suff_stats(phi_estimate_185))
# # L2_250 = L2dist(expected_suff_stats(phi_estimate_baseline), expected_suff_stats(phi_estimate_250))
#
# L2_dist = [L2_1, L2_60, L2_125, L2_185]
# print("L2 distance :", [L2_1, L2_60, L2_125, L2_185])

fairness_metrics = pd.read_pickle("fairness_metrics.pkl")
fairness_level = fairness_metrics[0]

x_axis = [fairness_level[0], fairness_level[59], fairness_level[124], fairness_level[184]]

# plt.figure(3)
# # plt.plot(x_axis, L2_dist)
# sns.set(style="darkgrid")
# ax = sns.regplot(x=x_axis, y=L2_dist, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
# ax = sns.lineplot(x=x_axis, y=L2_dist)
# # ax = sns.regplot(x=x_axis, y=L2_dist,
# #                  scatter_kws={"s": 80},
# #                  order=3, ci=None)
#
# ax.set_xlabel("fairness (% rule) in terms of race")
# ax.set_ylabel("L2 distance in sufficient stats")
# plt.savefig('L2dist_fair.pdf')


KLD_1 = KL_Dir(phi_estimate_baseline, phi_estimate_1)
KLD_60 = KL_Dir(phi_estimate_baseline, phi_estimate_60)
KLD_125 = KL_Dir(phi_estimate_baseline, phi_estimate_125)
KLD_185 = KL_Dir(phi_estimate_baseline, phi_estimate_185)

KLD = [KLD_1, KLD_60, KLD_125, KLD_185]

plt.figure(4)
ax = sns.regplot(x=x_axis, y=KLD, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
ax = sns.lineplot(x=x_axis, y=KLD)
ax.set_title('Q-FIT')
# ax = sns.regplot(x=x_axis, y=L2_dist,
#                  scatter_kws={"s": 80},
#                  order=3, ci=None)

ax.set_xlabel("fairness (% rule) in terms of race")
ax.set_ylabel("KLD")
plt.savefig('KLD_fair.pdf')


### now it's about INVASE
p_0 = [0.95181966, 0.16875832, 0.19484843, 0.15316528, 0.9802809,  0.9899059, 0.17358159, 0.9128857,  0.94334525, 0.3418799,  0.7686064,  0.1568285 ]
p_1 = [0.8964215,  0.15054446, 0.19511573, 0.17139837, 0.9775447,  0.9933273, 0.14426257, 0.78853, 0.9286467, 0.26026243, 0.74393994, 0.16311565]
p_60 = [0.88381696, 0.07678371, 0.11549377, 0.14349946, 0.9825279,  0.9949945, 0.1290174,  0.08738978, 0.9323417,  0.13599676, 0.60991037, 0.11625227]
p_125 = [0.88225114, 0.05454433, 0.04139346, 0.12868208, 0.98240954, 0.9926068, 0.09005944, 0.01443112, 0.9387729,  0.16173232, 0.51385397, 0.10628422]
p_185 = [0.94819814, 0.05639534, 0.03604005, 0.13409953, 0.97774965, 0.99215436, 0.07348135, 0.01271993, 0.9578264,  0.12007644, 0.45237222, 0.08673371]


KLD_1 = KL_Bern(p_0, p_1)
KLD_60 = KL_Bern(p_0, p_60)
KLD_125 = KL_Bern(p_0, p_125)
KLD_185 = KL_Bern(p_0, p_185)

KLD = [KLD_1, KLD_60, KLD_125, KLD_185]

plt.figure(5)
ax = sns.regplot(x=x_axis, y=KLD, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
ax = sns.lineplot(x=x_axis, y=KLD)
# ax = sns.regplot(x=x_axis, y=L2_dist,
#                  scatter_kws={"s": 80},
#                  order=3, ci=None)

ax.set_xlabel("fairness (% rule) in terms of race")
ax.set_ylabel("KLD")
ax.set_title("INVASE")

plt.savefig('KLD_fair_INVASE.pdf')



plt.figure(6)
# https://seaborn.pydata.org/examples/color_palettes.html
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,12)
sns.barplot(x=x, y=p_0, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("baseline")
ax1.set_ylim(0,0.55)
ax1.set_title('INVASE')

fairness_level = fairness_metrics[0].to_numpy()

T_iter = 1
sns.barplot(x=x, y=p_1, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter]
ax2.set_ylabel(str(np.int(level))+"% Fair")
ax2.set_ylim(0,0.55)

T_iter = 60

sns.barplot(x=x, y=p_60, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter-1]
ax3.set_ylabel(str(np.int(level))+"% Fair")
ax3.set_ylim(0,0.55)

T_iter = 125

sns.barplot(x=x, y=p_125, palette="rocket", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter]
ax4.set_ylabel(str(np.int(level))+"% Fair")
ax4.set_ylim(0,0.55)

T_iter = 185

sns.barplot(x=x, y=p_185, palette="rocket", ax=ax5)
ax5.axhline(0, color="k", clip_on=False)
level = fairness_level[T_iter-1]
ax5.set_ylabel(str(np.int(level))+"% Fair")
ax5.set_ylim(0,0.55)

plt.savefig('selection probability.pdf')


# ### L2 distance
#
# L2_1 = L2dist(p_0, p_1)
# L2_60 = L2dist(p_0, p_60)
# L2_125 = L2dist(p_0, p_125)
# L2_185 = L2dist(p_0, p_185)
#
# L2_dist=[L2_1, L2_60, L2_125, L2_185]
#
# plt.figure(6)
# ax = sns.regplot(x=x_axis, y=L2_dist, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
# ax = sns.lineplot(x=x_axis, y=L2_dist)
# # ax = sns.regplot(x=x_axis, y=L2_dist,
# #                  scatter_kws={"s": 80},
# #                  order=3, ci=None)
#
# ax.set_xlabel("fairness (% rule) in terms of race")
# ax.set_ylabel("L2 dist on Suff Stats")
# ax.set_title("INVASE")
#
# plt.savefig('Ss_fair_INVASE.pdf')