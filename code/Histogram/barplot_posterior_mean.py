
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def mean_then_normalize(v):
    v_mean = np.mean(v,0)
    # normalized_mean = v_mean/np.sum(v_mean)
    # return normalized_mean
    return v_mean


sns.set(font_scale=1.35)

# https://seaborn.pydata.org/examples/color_palettes.html
plt.figure(1)
# plt.title('Shapely Value')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 5), sharex=True)
x = np.arange(0,10)
#
# dataset = 'XOR'
# filename = dataset+'posterior_mean.npy'
# qfit = mean_then_normalize(np.load(filename))
#
# sns.barplot(x=x, y=qfit, palette="rocket", ax=ax1)
# ax1.axhline(0, color="k", clip_on=False)
# ax1.set_ylabel(dataset)
ax1.set_title("Importance by Q-FIT")

# dataset = 'orange_skin'
# filename = dataset+'posterior_mean.npy'
# qfit = mean_then_normalize(np.load(filename))
#
# sns.barplot(x=x, y=qfit, palette="rocket", ax=ax2)
# ax2.axhline(0, color="k", clip_on=False)
# ax2.set_ylabel("OS")


dataset = 'nonlinear_additive'
filename = dataset+'posterior_mean.npy'
qfit = mean_then_normalize(np.load(filename))

sns.barplot(x=x, y=qfit, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
# ax3.set_ylabel("NA")
ax1.set_xlabel("Features")

# ax1.set_title("importance by Q-FIT")
plt.savefig('qfit.pdf')



# https://seaborn.pydata.org/examples/color_palettes.html
# plt.figure(2)
# # plt.title('Shapely Value')
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
# x = np.arange(0,10)
#
# dataset = 'XOR'
# filename = dataset+'shap.npy'
# sv = mean_then_normalize(np.load(filename))
#
# sns.barplot(x=x, y=sv, palette="rocket", ax=ax1)
# ax1.axhline(0, color="k", clip_on=False)
# # ax1.set_ylabel("XOR")
# ax1.set_title("Shapely Value by SHAP")
#
# dataset = 'orange_skin'
# filename = dataset+'shap.npy'
# sv = mean_then_normalize(np.load(filename))
#
# sns.barplot(x=x, y=sv, palette="rocket", ax=ax2)
# ax2.axhline(0, color="k", clip_on=False)
# # ax2.set_ylabel("OS")

dataset = 'nonlinear_additive'
filename = dataset+'shap.npy'
sv = mean_then_normalize(np.load(filename))

sns.barplot(x=x, y=sv, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
# ax3.set_ylabel("NA")
ax2.set_xlabel("Features")

# plt.savefig('sv.pdf')

# INVASE

# plt.figure(3)
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
# x = np.arange(0,10)
#
# invase = [0.997, 0.998, 0.008, 0.012, 0.023, 0.026, 0.014, 0.012, 0.015, 0.009]
#
# sns.barplot(x=x, y=invase, palette="rocket", ax=ax1)
# ax1.axhline(0, color="k", clip_on=False)
# # ax1.set_ylabel("XOR")
# ax1.set_title("Selection probability by INVASE")
#
# invase = [1.,    1.,    1.,    1., 0.001, 0.001,    0.001, 0.001, 0.001, 0.001]
#
# sns.barplot(x=x, y=invase, palette="rocket", ax=ax2)
# ax2.axhline(0, color="k", clip_on=False)
# # ax2.set_ylabel("OS")

invase = [1.,    0.999, 0.999, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

sns.barplot(x=x, y=invase, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
# ax3.set_ylabel("NA")
ax3.set_xlabel("Features")

# plt.savefig('invase.pdf')

plt.savefig('NA.pdf')