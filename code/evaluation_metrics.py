""" this script provides evaluations metrics """

# """ how to use these evaluation metrics in practice """
#
# dataset = args.dataset
# S, datatypes_test_samp = test_instance(dataset, True, False)
# if dataset == "xor":
#     k = 2
# elif dataset == "orange_skin" or dataset == "nonlinear_additive" or dataset == "alternating":  # dummy for alternating
#     k = 4
# elif dataset == "syn4":
#     k = 7
# elif dataset == "syn5" or dataset == "syn6":
#     k = 8
#
# median_ranks = compute_median_rank(S, k, dataset, datatypes_test_samp)
# mean_median_ranks = np.mean(median_ranks)
# tpr, fdr = binary_classification_metrics(S, k, dataset, mini_batch_size, datatypes_test_samp)
# print("mean median rank", mean_median_ranks)
# print(f"tpr: {tpr}, fdr: {fdr}")
#
# # 1.5 - xor
# # 2.67 - orange_skin
# # 2.56 - nonlinear_additive
# # 2.88/ 3.5 - alternating


import numpy as np


#######################################
# evaluation

def create_rank(scores, k):
    # scores (100000, 10) #k - number opf ground truth relevant features, e.g. 2 or 4
    """
    Compute rank of each feature based on weight.

    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank = (-permutated_weights).argsort().argsort() + 1
        rank = permutated_rank[np.argsort(idx)]

        rank = np.argsort(score.detach().cpu().numpy())[::-1]


        if type(rank).__module__=='torch':
            ranks.append(rank.numpy())
        else:
            ranks.append(rank)

    return np.array(ranks)

##sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)[source]¶
from sklearn.metrics import matthews_corrcoef

def get_mmc(arr1, arr2):
    mcc = matthews_corrcoef(np.sort(arr1).flatten(), np.sort(arr2).flatten()) #2000,2 - 4000,
    #arr1 is [1,2], [1,2] [1,2], etc.
    #arr2 is [1,2], [1,2] [1,2]
    return mcc


def get_tpr(arr1, arr2):
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def difference(arr1, arr2):
        diff = list(set(arr2) - set(arr1))
        return diff

    all = 0
    all_tp = 0
    fp = 0
    for i in range(arr1.shape[0]):
        tp = intersection(arr1[i], arr2[i])
        diff = difference(arr1[i], arr2[i])

        fp += len(diff)
        all_tp += len(tp)
        all += len(arr1[i])

    fdr = float(fp) / (fp + all_tp)
    tpr = float(all_tp) / all
    return tpr, fdr


'''
by 'positions' we mean the rank of a feature, best feature position = 1
gtfeatures_positions - ordered position numbers for k relevant features, e.g. [1,2,3,4,5] for k=5
switch_gtfeatures_positions - positions of the gtfeatures as indicated by switches, ideally set (gtfeatures_positions) = set (switch_gtfeatures_positions)

by 'features' we mean the numbers corresponding to ordered features
gtfeatures - features that generated the label, e.g. for alternating it is 1,2,3,4,10 or 5,6,7,8, 10 (starting form 1) .
'''

def binary_classification_metrics(scores, k, dataset, mini_batch_size, datatype_val=None, instancewise=False):
    tpr, fdr = 0, 0
    ranks = create_rank(scores, k)  # ranks start with 1 and end with 10 (not 0 to 9),
    # [7,6,1,4,3,5,8,11,9,10,2] means feature 7 is the smallest and 2 the biggest
    if dataset == "xor" or dataset == "orange_skin" or dataset == "nonlinear_additive":

        # gt features positiions
        gtfeatures_positions = np.tile(np.arange(k), (mini_batch_size, 1))
        onehots_arr_gt = np.zeros_like(scores.detach().cpu().numpy())
        for i, elem in enumerate(onehots_arr_gt):
            elem[gtfeatures_positions[i]] = 1


        # qfit features positions
        if  instancewise:
            switch_gtfeatures_positions = ranks[:, :k]  # (mini_batch_size, k)
            onehots_arr=np.zeros_like(scores.detach().cpu().numpy())
            for i, elem in enumerate(onehots_arr):
                elem[switch_gtfeatures_positions[i]]=1
        else:
            onehots_arr=np.zeros_like(scores.detach().cpu().numpy())




    elif dataset == "alternating":

        gtfeatures_positions = np.tile(np.arange(k) + 1, (mini_batch_size, 1))

        datatype_val = datatype_val[:len(scores)]
        gtfeatures = np.dstack([(datatype_val == 'orange_skin')] * 5) * np.array([1, 2, 3, 4, 10])
        gtfeatures[0][datatype_val == 'nonlinear_additive'] = np.array([5, 6, 7, 8, 10])
        gtfeatures = gtfeatures[0]

        switch_gtfeatures_positions = []
        for i in range(mini_batch_size):
            switch_gtfeatures_position = ranks[i][gtfeatures[i] - 1]
            switch_gtfeatures_positions.append(switch_gtfeatures_position)
        switch_gtfeatures_positions = np.array(switch_gtfeatures_positions)

    elif "syn" in dataset:
        gtfeatures = datatype_val + 1         #adjusting the ground truth features

        gtfeatures_positions = [] # indices of gt features
        switch_gtfeatures_positions = []
        for i in range(gtfeatures.shape[0]): # for each data sample
            switch_gtfeatures_positions.append(ranks[i][gtfeatures[i] - 1]) # getting the
            gtfeatures_positions.append(np.arange(len(gtfeatures[i])) + 1)
        switch_gtfeatures_positions = np.array(switch_gtfeatures_positions)
        gtfeatures_positions = np.array(gtfeatures_positions) # these are indices in the array which should be best (+1)
        important_features_num = [len(i) for i in gtfeatures_positions]

        onehots_arr_gt = np.zeros_like(scores.detach().cpu().numpy())
        for i, elem in enumerate(onehots_arr_gt):
            elem[datatype_val[i]] = 1

        onehots_arr = np.zeros_like(scores.detach().cpu().numpy())
        for i, elem in enumerate(onehots_arr):
            elem[ranks[i, :important_features_num[i]]] = 1

    tpr, fdr = get_tpr(gtfeatures_positions, switch_gtfeatures_positions)

    # for i in switch_gtfeatures_positions:
    #     if 11 in i:
    #         print("ll")

    # gtfeatures_positions_arr=[]
    # for ind, dat in enumerate(gtfeatures_positions):
    #     gtfeatures_positions_arr.append(onehot(dat, 11))
    #
    # switch_gtfeatures_positions_arr=[]
    # for ind, dat in enumerate(switch_gtfeatures_positions):
    #     switch_gtfeatures_positions_arr.append(onehot(dat, 11))

    #mcc = matthews_corrcoef(np.array(gtfeatures_positions_arr).flatten(), np.array(switch_gtfeatures_positions_arr).flatten())
    mcc = matthews_corrcoef(onehots_arr.flatten(), onehots_arr_gt.flatten())

    return tpr, fdr, mcc

def onehot(inds, n):
    print(n)
    a=np.zeros(n)
    print(inds)
    a[inds]=1
    return a


def compute_median_rank(scores, k, dataset, datatype_val=None):
    ranks = create_rank(scores, k)+1
    if dataset == "xor" or dataset == "orange_skin" or dataset == "nonlinear_additive":
        median_ranks = np.median(ranks[:, :k], axis=1)
    elif dataset == "alternating":
        datatype_val = datatype_val[:len(scores)]
        # [datatype_val == 'orange_skin', :] is 1 for orange skin in ranks 1st dim, and 0 otherwise
        median_ranks1 = np.median(ranks[datatype_val == 'orange_skin', :][:, np.array([0, 1, 2, 3, 9])],
                                  axis=1)
        median_ranks2 = np.median(
            ranks[datatype_val == 'nonlinear_additive', :][:, np.array([4, 5, 6, 7, 9])], axis=1)
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    elif dataset == "syn4" or dataset == "syn5" or dataset == "syn6":
        median_ranks_arr = []
        for i, data in enumerate(datatype_val):
            median_ranks_arr.append(np.median(ranks[i, datatype_val[i]]))

        median_ranks = np.array(median_ranks_arr)
    else:
        median_ranks = -1

    return median_ranks



