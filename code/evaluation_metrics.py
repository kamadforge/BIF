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

        ranks.append(rank.numpy())

    return np.array(ranks)


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
relevant_features_gt_positions - ordered position numbers for k relevant features, e.g. [1,2,3,4,5] for k=5
switch_relevant_features_positions - 
'''

def binary_classification_metrics(scores, k, dataset, mini_batch_size, datatype_val=None):
    tpr, fdr = 0, 0
    ranks = create_rank(scores, k)  # ranks start with 1 and end with 10 (not 0 to 9)

    if dataset == "xor" or dataset == "orange_skin" or dataset == "nonlinear_additive":

        relevant_features_gt_positions = np.tile(np.arange(k) + 1, (mini_batch_size, 1))

        switch_relevant_features_positions = ranks[:, :k]  # (mini_batch_size, k)


    elif dataset == "alternating":

        relevant_features_gt_positions = np.tile(np.arange(k) + 1, (mini_batch_size, 1))

        datatype_val = datatype_val[:len(scores)]
        relevant_features = np.dstack([(datatype_val == 'orange_skin')] * 5) * np.array([1, 2, 3, 4, 10])
        relevant_features[0][datatype_val == 'nonlinear_additive'] = np.array([5, 6, 7, 8, 10])
        relevant_features = relevant_features[0]

        switch_relevant_features_positions = []
        for i in range(mini_batch_size):
            switch_relevant_features_position = ranks[i][relevant_features[i] - 1]
            switch_relevant_features_positions.append(switch_relevant_features_position)
        switch_relevant_features_positions = np.array(switch_relevant_features_positions)

    elif "syn" in dataset:
        relevant_features = datatype_val + 1

        relevant_features_gt_positions = []
        switch_relevant_features_positions = []
        for i in range(relevant_features.shape[0]):
            switch_relevant_features_positions.append(ranks[i][relevant_features[i] - 1])
            relevant_features_gt_positions.append(np.arange(len(relevant_features[i])) + 1)
        switch_relevant_features_positions = np.array(switch_relevant_features_positions)
        relevant_features_gt_positions = np.array(relevant_features_gt_positions)

    tpr, fdr = get_tpr(relevant_features_gt_positions, switch_relevant_features_positions)

    return tpr, fdr


def compute_median_rank(scores, k, dataset, datatype_val=None):
    ranks = create_rank(scores, k)
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

    return median_ranks



