import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from distance_Dirichlet import L2dist, expected_suff_stats, KL_Dir, KL_Bern

# sigma = 68.7 for eps = 0.01
# sigma = 8.8 for eps = 0.1
# sigma = 2.4 for eps = 1.0
# sigma = 0.84 for eps = 4.0
# sigma = 1e-6 for eps = infty (nonprivate)

noise_level = [1e-6, 0.84, 2.4, 8.8, 68.7]

""" first plot : accuracy vs privacy """
# maxseed = 1
# maxepoch = 20
# maxnoise = len(privacy_level) # including sigma = 0
#
# ROC = np.zeros((maxseed, maxepoch, maxnoise))
# phi_estimate_all =np.zeros(())
#
# for seednum in range(0, maxseed):
#
#     for noise_level in range(0, maxnoise):
#         if noise_level==0:
#             dp_sigma=0.0
#         elif noise_level==1:
#             dp_sigma=1.0
#         elif noise_level==2:
#             dp_sigma = 2.0
#         elif noise_level==3:
#             dp_sigma = 4.0
#         elif noise_level==4:
#             dp_sigma = 8.0
#         else:
#             dp_sigma = 17.0
#         filename = 'pri_' + str(dp_sigma) + 'seed_' + str(seednum) + 'roc.npy'
#         roc = np.load(filename)
#         print(filename)
#         print(roc)
#         ROC[seednum,:,noise_level] = np.load(filename)
#
# mean_ROC = np.mean(ROC[:,19,:], 0)
# std_ROC = np.std(ROC[:,19,:],0)
#
# sns.set(style="whitegrid")
# plt.figure(1)
# # plt.plot(fairness_metrics["race"], eval_metrics["ROC AUC"])
# # ax = sns.lineplot(x=privacy_level, y=mean_ROC, ci=std_ROC, palette="tab10", linewidth=2.5)
# # ax = sns.errorbar()
# plt.errorbar(privacy_level, mean_ROC, yerr=std_ROC, fmt='o', linewidth=2.5, color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0)
# plt.xlabel('Privacy in terms of noise level')
# plt.ylabel('ROC AUC')
# plt.savefig('privacy_vs_roc.pdf')


""" second plot : KLD vs privacy """
# unpack all the results
input_dim  = 22
# mean_importance_all = np.zeros((len(noise_level), input_dim))
phi_est_mat_all = np.zeros((len(noise_level), input_dim))
seednum = 0

for nse_lv in range(0, len(noise_level)):

    # filename =  'pri_' + str(privacy_level[nse_lv]) + 'seed_' + str(seednum) + 'importance.npy'
    # mean_importance_all[nse_lv, seednum, :] = np.mean(np.load(filename), axis=0)
    filename = 'pri_' + str(noise_level[nse_lv]) + 'seed_' + str(seednum) + 'phi_est.npy'
    phi_est_mat_all[nse_lv, :] = np.load(filename)

KLD = np.zeros(len(noise_level)-1)
KLD[0] = KL_Dir(phi_est_mat_all[0,:], phi_est_mat_all[1,:])
KLD[1] = KL_Dir(phi_est_mat_all[0,:], phi_est_mat_all[2,:])
KLD[2] = KL_Dir(phi_est_mat_all[0,:], phi_est_mat_all[3,:])
KLD[3] = KL_Dir(phi_est_mat_all[0,:], phi_est_mat_all[4,:])
# for nse_lv in range(0, len(noise_level)-1):
#     KLD[nse_lv] = KL_Dir(phi_est_mat_all[0,:], phi_est_mat_all[nse_lv+1,:])

print('KLD ', KLD)

# # KLD = [KLD_1, KLD_60, KLD_125, KLD_185, KLD_250]
# #
# KLD = np.median(KLD, axis=1)
# plt.figure(4)
# ax = sns.regplot(x=privacy_level[1:], y=KLD, fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200})
# ax = sns.lineplot(x=privacy_level[1:], y=KLD)
# ax.set_title('Q-FIT')
#
# ax.set_xlabel("privacy in terms of noise level")
# ax.set_ylabel("KLD")
# plt.savefig('KLD_privacy.pdf')
