
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def mean_then_normalize(v):
    v_mean = np.mean(v,0)
    normalized_mean = v_mean/sum(v_mean)
    return normalized_mean


# load results
dataset = 'nonlinear_additive'
filename = dataset+'posterior_mean.npy'
qfit = mean_then_normalize(np.load(filename))
# mean_five_run = np.mean(five_run,0)
# normalized_mean = mean_five_run/sum(mean_five_run)

# https://seaborn.pydata.org/examples/color_palettes.html
plt.figure(1)
# plt.title('Shapely Value')
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,10)

sns.barplot(x=x, y=qfit, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel(dataset)
ax1.set_title("Q-FIT")


sns.set(font_scale=1.3)

# https://seaborn.pydata.org/examples/color_palettes.html
plt.figure(1)
# plt.title('Shapely Value')
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,10)

dataset = 'XOR'
filename = dataset+'shap.npy'
sv = np.load(filename)

sns.barplot(x=x, y=sv, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("XOR")
ax1.set_title("SHAP")

dataset = 'orange_skin'
filename = dataset+'shap.npy'
sv = np.load(filename)

sns.barplot(x=x, y=sv, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("OS")


dataset = 'nonlinear_additive'
filename = dataset+'shap.npy'
sv = np.load(filename)

sns.barplot(x=x, y=sv, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("NA")
ax3.set_xlabel("features")


plt.savefig('sv.pdf')
#
# plt.show()


# plt.savefig('learned_importance.pdf')


plt.figure(2)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
x = np.arange(0,10)

dataset = 'XOR'
filename = dataset+'posterior_mean.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_title("importance by Q-FIT")
ax1.set_ylabel("XOR")

dataset = 'orange_skin'
filename = dataset+'posterior_mean.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("OS")

dataset = 'nonlinear_additive'
filename = dataset+'posterior_mean.npy'
mean_importance = np.load(filename)

sns.barplot(x=x, y=mean_importance, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("NA")
ax3.set_xlabel("features")

plt.savefig('qfit.pdf')

############################

#
# plt.show()