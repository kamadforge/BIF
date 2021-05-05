import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from distance_Dirichlet import KL_Dir, KL_Bern
import matplotlib

## plotting the results
font = {
    # 'family': 'normal',
    # 'weight': 'bold',
    'size': 14}

matplotlib.rc('font', **font)

# sigma = 68.7 for eps = 0.01
# sigma = 8.8 for eps = 0.1
# sigma = 2.4 for eps = 1.0
# sigma = 0.84 for eps = 4.0
# sigma = 1e-6 for eps = infty (nonprivate)

noise_level = [1e-6, 0.84, 2.4, 8.8, 68.7]
privacy_level = [1e6, 4.0, 1.0, 0.1, 0.01] # in terms of epsilon

""" first plot : accuracy vs privacy """
ROC = np.zeros(len(noise_level))
for nse_lv in range(0, len(noise_level)):
    filename = 'pri_' + str(noise_level[nse_lv]) + 'seed_' + str(0) + 'roc.npy'
    ROC[nse_lv] = np.load(filename)
sns.set(style="whitegrid")
plt.figure(1)
plt.subplot(131)
plt.plot(privacy_level, ROC, 'o-')
plt.xticks(privacy_level)
plt.xscale('log')
plt.xlabel('Privacy (eps)')
plt.ylabel('ROC')
# plt.show()
plt.savefig('privacy_vs_roc_diabetic.pdf')


""" second plot : KLD vs privacy """
# unpack all the results
input_dim  = 22
mean_importance_all = np.zeros((len(noise_level), input_dim))
phi_est_mat_all = np.zeros((len(noise_level), input_dim))
seednum = 0

for nse_lv in range(0, len(noise_level)):

    filename =  'pri_' + str(noise_level[nse_lv]) + 'seed_' + str(seednum) + 'importance.npy'
    mean_importance_all[nse_lv, :] = np.load(filename)
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
sns.set(style="whitegrid")
plt.subplot(132)
plt.plot(privacy_level[1:], KLD, 'o-')
plt.xticks(privacy_level[1:])
plt.xscale('log')
plt.xlabel('Privacy (eps)')
plt.ylabel('KLD (BIF)')
# plt.savefig('privacy_vs_KLD_BIF_diabetic.pdf')

##### subplot(133) for INVASE ####




""" third plot: learned importance in each case """
top_few = [0, 1, 2, 3, 4, 5, 6]
column_names = np.load('column_names_diabetic.npy', allow_pickle=True)


plt.figure(3)
plt.subplot(511)
mean_importance = mean_importance_all[0, :]
order_by_importance = np.argsort(mean_importance)[::-1] # descending order
sns.barplot(y=[element for element in column_names[order_by_importance][top_few]],
            x=[element for element in mean_importance[order_by_importance][top_few]])
plt.xlim([0,0.6])
plt.title('BIF (non_priv)')

# privacy_level = [1e6, 4.0, 1.0, 0.1, 0.01] # in terms of epsilon

plt.subplot(512)
mean_importance = mean_importance_all[1, :]
plt.title('BIF (eps=%.2f)'%privacy_level[1])
order_by_importance = np.argsort(mean_importance)[::-1] # descending order
sns.barplot(y=[element for element in column_names[order_by_importance][top_few]],
            x=[element for element in mean_importance[order_by_importance][top_few]])
plt.xlim([0,0.6])

plt.subplot(513)
mean_importance = mean_importance_all[2, :]
plt.title('BIF (eps=%.2f)'%privacy_level[2])
order_by_importance = np.argsort(mean_importance)[::-1] # descending order
sns.barplot(y=[element for element in column_names[order_by_importance][top_few]],
            x=[element for element in mean_importance[order_by_importance][top_few]])
plt.xlim([0,0.6])

plt.subplot(514)
mean_importance = mean_importance_all[3, :]
plt.title('BIF (eps=%.2f)'%privacy_level[3])
order_by_importance = np.argsort(mean_importance)[::-1] # descending order
sns.barplot(y=[element for element in column_names[order_by_importance][top_few]],
            x=[element for element in mean_importance[order_by_importance][top_few]])
plt.xlim([0,0.6])

plt.subplot(515)
mean_importance = mean_importance_all[4, :]
plt.title('BIF (eps=%.2f)'%privacy_level[4])
order_by_importance = np.argsort(mean_importance)[::-1] # descending order
sns.barplot(y=[element for element in column_names[order_by_importance][top_few]],
            x=[element for element in mean_importance[order_by_importance][top_few]])
plt.xlim([0,0.6])


plt.show()

