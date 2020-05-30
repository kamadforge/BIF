"""
This script is written for visualizing the results of Adult data for fairness
"""

__author__ = 'mijung'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

""" first plot : accuracy vs fairness"""
# load metrics
# eval_metrics = np.load('eval_metrics.npy') # first column: ROC, second column: Accuracy
# fairness_metrics = np.load('fairness_metrics.npy') # first column: Race, second column: Sex


# fairness_metrics = pd.read_pickle("fairness_metrics.pkl")
# eval_metrics = pd.read_pickle("eval_metrics.pkl")

# sns.set(style="whitegrid")
# plt.subplot(211)
# plt.figure(1)
# plt.plot(fairness_metrics[0].to_numpy(), eval_metrics[0].to_numpy())
# ax = sns.lineplot(fairness_metrics[0].to_numpy(), eval_metrics[0].to_numpy(), palette="tab10", linewidth=2.5)
# ax.set(xlabel='Fairness in terms of Race', ylabel='ROC')
# plt.savefig('images/fairness_vs_roc.pdf')
# plt.subplot(212)
# sns.lineplot(fairness_metrics["sex"], eval_metrics["Accuracy"], palette="tab10", linewidth=2.5)



""" second plot : distance of feature importance vs fairness when baseline, T=1, 125 and 250 """
mean_importance_baseline = np.load('baseline_importance.npy')
phi_estimate_baseline = np.load('baseline_phi_est.npy')

# https://seaborn.pydata.org/examples/color_palettes.html
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,12)
sns.barplot(x=x, y=mean_importance_baseline, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("baseline importance")

T_iter = 1
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
filename = 'fair_clf_' + str(T_iter) + 'phi_est.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("importance (T=1)")

T_iter = 125
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
filename = 'fair_clf_' + str(T_iter) + 'phi_est.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("importance (T=125)")

T_iter = 250
filename = 'fair_clf_' + str(T_iter) + 'importance.npy'
mean_importance = np.load(filename)
filename = 'fair_clf_' + str(T_iter) + 'phi_est.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.set_ylabel("importance (T=250)")

plt.savefig('learned_importance.pdf')

""" thrid plot : learned features when T=1, 125 and 250 """






