"""
This script is written for visualizing the results of Adult data
"""

__author__ = 'anon_m'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

dataset = "adult"

# load our results
# iter_sigmas = np.array([0., 1., 10., 50., 100.])
iter_sigmas = np.array([0.])

for k in range(iter_sigmas.shape[0]):

    switch_posterior_mean = np.load('weights/%s_switch_posterior_mean' % dataset + str(int(iter_sigmas[k]))+'.npy')


    ##############################
    # age = switch_posterior_mean[:,0]
    # workclass = switch_posterior_mean[:,1]
    # fnlwgt = switch_posterior_mean[:,2]
    # education = switch_posterior_mean[:,3]
    # education_num = switch_posterior_mean[:,4]
    #
    # marital_status = switch_posterior_mean[:,5]
    # occupation = switch_posterior_mean[:,6]
    # relationship = switch_posterior_mean[:,7]
    # race = switch_posterior_mean[:,8]
    # sex = switch_posterior_mean[:,9]
    #
    # capital_gain = switch_posterior_mean[:,10]
    # capital_loss = switch_posterior_mean[:,11]
    # hours_per_week = switch_posterior_mean[:,12]
    # native_country = switch_posterior_mean[:,13]

    # random_dists = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    #                 'marital_status', 'occupation', 'relationship', 'race', 'sex',
    #                 'capital_gain', 'capical_loss', 'hours_per_week', 'native_country']



    samp, featnum= switch_posterior_mean.shape

    random_dists = ["f_%i" % i for i in range(featnum)]


    N = samp

    switch_posterior_mean_t=switch_posterior_mean.transpose()
    data=switch_posterior_mean_t.tolist()

    #
    # data = [
    #     age, workclass, fnlwgt, education, education_num,
    #     marital_status, occupation, relationship, race, sex,
    #     capital_gain, capital_loss, hours_per_week, native_country]

    ###############################33333

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Adult data')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Bayesian Logistic regression with Adult data with sigma='+str(int(iter_sigmas[k])))
    ax1.set_xlabel('Different noise level')
    ax1.set_ylabel('AUC on test data')

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(4):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        if i<4:
            ax1.add_patch(Polygon(box_coords, facecolor=box_colors[1]))
        else:
            ax1.add_patch(Polygon(box_coords, facecolor=box_colors[0]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        # for j in range(2):
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(random_dists, 1),
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    # upper_labels = [str(np.round(s, 1)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        # k = tick % 2
        # k = tick
        if tick<4:
            ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[1], color=box_colors[1])
        else:
            ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[1], color=box_colors[0])

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, 'average over 20 random initalizations',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    fig.text(0.80, 0.08, '*', color='white', backgroundcolor='silver',
             weight='roman', size='medium')
    fig.text(0.815, 0.085, ' Average Value', color='black', weight='roman',
             size='x-small')

    plt.show()

    filename = 'posterior mean of switches with sigma=' + str(int(iter_sigmas[k]))
    fig.savefig(filename)