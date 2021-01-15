"""
This script is written for visualizing the results of Adult data for privacy
"""

__author__ = 'mijung'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from distance_Dirichlet import L2dist, expected_suff_stats, KL_Dir, KL_Bern


privacy_level = [0.0, 1.0, 2.0, 4.0, 8.0, 17.0]

""" first plot : accuracy vs privacy """
# for baseline, seednum = range(0, 10)
maxseed = 5
maxepoch = 20
maxnoise = len(privacy_level) # including sigma = 0

ROC = np.zeros((maxseed, maxepoch, maxnoise))
phi_estimate_all =np.zeros(())

for seednum in range(0, maxseed):

    for noise_level in range(0, maxnoise):
        if noise_level==0:
            dp_sigma=0.0
        elif noise_level==1:
            dp_sigma=1.0
        elif noise_level==2:
            dp_sigma = 2.0
        elif noise_level==3:
            dp_sigma = 4.0
        elif noise_level==4:
            dp_sigma = 8.0
        else:
            dp_sigma = 17.0
        filename = 'pri_' + str(dp_sigma) + 'seed_' + str(seednum) + 'roc.npy'
        roc = np.load(filename)
        print(filename)
        print(roc)
        ROC[seednum,:,noise_level] = np.load(filename)

mean_ROC = np.mean(ROC[:,19,:], 0)
std_ROC = np.std(ROC[:,19,:],0)

sns.set(style="whitegrid")
plt.figure(1)
# plt.plot(fairness_metrics["race"], eval_metrics["ROC AUC"])
# ax = sns.lineplot(x=privacy_level, y=mean_ROC, ci=std_ROC, palette="tab10", linewidth=2.5)
# ax = sns.errorbar()
plt.errorbar(privacy_level, mean_ROC, yerr=std_ROC, fmt='o', linewidth=2.5, color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.xlabel('Privacy in terms of noise level')
plt.ylabel('ROC AUC')
plt.savefig('privacy_vs_roc.pdf')


# unpack all the results
input_dim  = 14
seedmax_per_seed = 10
mean_importance_all = np.zeros((len(privacy_level), maxseed, input_dim))
phi_est_mat_all = np.zeros((len(privacy_level), maxseed, input_dim))


for seednum in range(0, maxseed):

    for nse_lv in range(0, len(privacy_level)):

        filename =  'pri_' + str(privacy_level[nse_lv]) + 'seed_' + str(seednum) + 'importance.npy'
        mean_importance_all[nse_lv, seednum, :] = np.mean(np.load(filename), axis=0)

        filename = 'pri_' + str(privacy_level[nse_lv]) + 'seed_' + str(seednum) + 'phi_est.npy'
        phi_est_mat_all[nse_lv, seednum, :] = np.mean(np.load(filename), axis=0)



mean_importance = np.median(mean_importance_all, axis=1)
# phi_est = np.median(phi_est_mat_all, axis=1)


# """ second plot : learned features when privacy_level = [0., 1.35, 2.3, 4.4, 8.4, 17.] """
plt.figure(2)
# https://seaborn.pydata.org/examples/color_palettes.html
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(7, 5), sharex=True)
ylim_max = 0.6

# seednum = 0
# mean_importance_baseline = np.zeros((maxseed, input_dim))
# for seednum in range(0, maxseed):
# filename = 'pri_' + str(0.0) + 'seed_' + str(seednum) + 'importance.npy'
# mean_importance_baseline = np.load(filename)
# mean_importance = np.mean(np.mean(mean_importance_all,0)

x = np.arange(0,input_dim)

T = 0
sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("baseline")
ax1.set_ylim(0,ylim_max)


# privacy_level = [0., 1.35, 2.3, 4.4, 8.4, 17.]
T = 1
level = privacy_level[T]
sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("noise:" + str(np.int(level)))
ax2.set_ylim(0,ylim_max)

T = 2
level = privacy_level[T]

sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("noise:" + str(np.int(level)))
ax3.set_ylim(0,ylim_max)

T = 3
level = privacy_level[T]

sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.set_ylabel("noise:" + str(np.int(level)))
ax4.set_ylim(0,ylim_max)

T = 4
level = privacy_level[T]

sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax5)
ax5.axhline(0, color="k", clip_on=False)
ax5.set_ylabel("noise:" + str(np.int(level)))
ax5.set_ylim(0,ylim_max)

T = 5
level = privacy_level[T]

sns.barplot(x=x, y=mean_importance[T,:], palette="rocket", ax=ax6)
ax6.axhline(0, color="k", clip_on=False)
ax6.set_ylabel("noise:" + str(np.int(level)))
ax6.set_ylim(0,ylim_max)


plt.savefig('learned_importance_vs_privacy.pdf')


""" third plot : distance of feature importance vs fairness """
# run code to learn phi for baseline, T=60,125,185,250 multiple times
# then compute the distance for each seed, then show the errorbar in this plot.
#
# T = 0
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_0 = np.mean(phi_estimate,0)
#
# T = 1
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_1 = np.mean(phi_estimate,0)
#
# T = 2
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_2 = np.mean(phi_estimate,0)
#
#
# T = 3
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_3 = np.mean(phi_estimate,0)
#
# T = 4
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_4 = np.mean(phi_estimate,0)
#
# T = 5
# level = privacy_level[T]
# # phi_estimate = np.zeros((maxseed, input_dim))
# # for seednum in range(0, maxseed):
# filename = 'pri_' + str(level) + 'seed_' + str(seednum) + 'phi_est.npy'
# phi_estimate = np.load(filename)
# phi_estimate_5 = np.mean(phi_estimate,0)

# # L2distance in suff stats
# L2_1 = L2dist(expected_suff_stats(phi_estimate_0), expected_suff_stats(phi_estimate_1))
# L2_60 = L2dist(expected_suff_stats(phi_estimate_0), expected_suff_stats(phi_estimate_2))
# L2_125 = L2dist(expected_suff_stats(phi_estimate_0), expected_suff_stats(phi_estimate_3))
# L2_185 = L2dist(expected_suff_stats(phi_estimate_0), expected_suff_stats(phi_estimate_4))
# L2_250 = L2dist(expected_suff_stats(phi_estimate_0), expected_suff_stats(phi_estimate_5))
#
# L2_dist = [L2_1, L2_60, L2_125, L2_185, L2_250]
# print("L2 distance :", [L2_1, L2_60, L2_125, L2_185, L2_250])

KLD = np.zeros((len(privacy_level)-1, maxseed))
for iter in range(0,maxseed):
    for nse_lv in range(0, len(privacy_level)-1):
        KLD[nse_lv, iter] = KL_Dir(np.mean(phi_est_mat_all[0,:,:], axis=0), phi_est_mat_all[nse_lv+1,iter,:])


# print('KLD ', KLD)

# KLD = [KLD_1, KLD_60, KLD_125, KLD_185, KLD_250]
#
KLD = np.median(KLD, axis=1)
plt.figure(4)
ax = sns.regplot(x=privacy_level[1:], y=KLD, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
ax = sns.lineplot(x=privacy_level[1:], y=KLD)
ax.set_title('Q-FIT')

ax.set_xlabel("privacy in terms of noise level")
ax.set_ylabel("KLD")
plt.savefig('KLD_privacy.pdf')


# # ### now it's about INVASE
#
# p_0 = [0.01522497, 0.00404595, 0.00404232, 0.00430544, 0.3845212,  0.9982528, 0.0050326,  0.00647169, 0.00339587, 0.00508895, 0.3117688,  0.00551459, 0.00711767, 0.00619385]
# p_1 = [0.02722446, 0.00546721, 0.00393242, 0.0037672,  0.26145563, 0.9965899, 0.00307376, 0.00665953, 0.00376468, 0.00373755, 0.15638779, 0.00971019, 0.00796957, 0.00363668]
# p_2 = [0.01100851, 0.00524833, 0.00516713, 0.00461853, 0.2879011,  0.9962535, 0.00342927, 0.01349588, 0.00446198, 0.00804025, 0.15700175, 0.07720158, 0.00817873, 0.00457497]
# p_3 = [0.01324397, 0.00435256, 0.00476793, 0.00296467, 0.22622485, 0.99748605,
#  0.00309202, 0.00929875, 0.0041021,  0.00707222, 0.16409422, 0.00703711,
#  0.00406151, 0.00485717]
# p_4 = [0.07897393, 0.00243676, 0.0031172,  0.00394712, 0.3338227,  0.99149746,
#  0.00243844, 0.00334673, 0.00295835, 0.00541442, 0.15128183, 0.0052301,
#  0.00793131, 0.0035188 ]
# p_5 = [0.47010514, 0.00236786, 0.00368441, 0.00691241, 0.47090974, 0.12390914, 0.00563126, 0.00411554, 0.00407621, 0.02614028, 0.15349102, 0.0232556, 0.01032341, 0.00436073]
#
# KLD_1 = KL_Bern(p_0, p_1)
# KLD_60 = KL_Bern(p_0, p_2)
# KLD_125 = KL_Bern(p_0, p_3)
# KLD_185 = KL_Bern(p_0, p_4)
# KLD_250 = KL_Bern(p_0, p_5)
#
# KLD = [KLD_1, KLD_60, KLD_125, KLD_185, KLD_250]
#
# plt.figure(5)
# x_axis = privacy_level[1:]
# ax = sns.regplot(x=x_axis, y=KLD, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
# ax = sns.lineplot(x=x_axis, y=KLD)
# # ax = sns.regplot(x=x_axis, y=L2_dist,
# #                  scatter_kws={"s": 80},
# #                  order=3, ci=None)
#
# ax.set_xlabel("privacy in terms of noise level")
# ax.set_ylabel("KLD")
# ax.set_title("INVASE")
#
# plt.savefig('KLD_privacy_INVASE.pdf')
# #


# plt.figure(6)
# y_lim_max = 1.0
# # https://seaborn.pydata.org/examples/color_palettes.html
# f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(7, 5), sharex=True)
# x = np.arange(0,input_dim)
# sns.barplot(x=x, y=p_0, palette="rocket", ax=ax1)
# ax1.axhline(0, color="k", clip_on=False)
# ax1.set_ylabel("baseline")
# ax1.set_ylim(0,y_lim_max)
# ax1.set_title('INVASE')
#
# T_iter = 1
# sns.barplot(x=x, y=p_1, palette="rocket", ax=ax2)
# ax2.axhline(0, color="k", clip_on=False)
# level = privacy_level[T_iter]
# ax2.set_ylabel("noise:"+str(np.int(level)))
# ax2.set_ylim(0,y_lim_max)
#
# T_iter = 2
# sns.barplot(x=x, y=p_2, palette="rocket", ax=ax3)
# ax3.axhline(0, color="k", clip_on=False)
# level = privacy_level[T_iter]
# ax3.set_ylabel("noise:"+str(np.int(level)))
# ax3.set_ylim(0,y_lim_max)
#
# T_iter = 3
# sns.barplot(x=x, y=p_3, palette="rocket", ax=ax4)
# ax4.axhline(0, color="k", clip_on=False)
# level = privacy_level[T_iter]
# ax4.set_ylabel("noise:"+str(np.int(level)))
# ax4.set_ylim(0,y_lim_max)
#
# T_iter = 4
# sns.barplot(x=x, y=p_4, palette="rocket", ax=ax5)
# ax5.axhline(0, color="k", clip_on=False)
# level = privacy_level[T_iter]
# ax5.set_ylabel("noise:"+str(np.int(level)))
# ax5.set_ylim(0,y_lim_max)
#
# T_iter = 5
# sns.barplot(x=x, y=p_5, palette="rocket", ax=ax6)
# ax6.axhline(0, color="k", clip_on=False)
# level = privacy_level[T_iter]
# ax6.set_ylabel("noise:"+str(np.int(level)))
# ax6.set_ylim(0,y_lim_max)
#
# plt.savefig('selection_probability_privacy.pdf')



# [0.00132304 0.00114428 0.00107561 0.00120788 0.15170467 0.00142995
#  0.00123394 0.8261998  0.0009765  0.00108621 0.00892246 0.00107646
#  0.00137722 0.00124191]

#  [0.12921901 0.00081508 0.00102447 0.00099089 0.25703073 0.00141194
#  0.00117411 0.44349247 0.00119146 0.00187413 0.1580977  0.00117545
#  0.00139458 0.00110808]

# [0.00226536 0.00115521 0.00114338 0.00132937 0.2568737  0.00169912
#  0.00133468 0.61061525 0.00105941 0.00132701 0.11628324 0.0014048
#  0.0022669  0.0012425 ]
