"""
This script contains functions for generating synthetic data. 
Copied from https://github.com/Jianbo-Lab/L2X/blob/master/synthetic/make_data.py
""" 


from __future__ import print_function
import numpy as np
from scipy.stats import chi2

def generate_XOR_labels(X):

    # sign = lambda a : (a>0)*1 - (a<0)*1
    #
    #
    # mult = X[:,0]*X[:,1]
    # signs= sign(mult)
    #
    # y = np.exp(signs*np.power(np.abs(X[:,0]*X[:,1]),1))
    #
    # #y = np.exp(X[:, 0] * X[:, 1])
    #
    # prob_1 = np.expand_dims(1 / (1+y) ,1)
    # prob_0 = np.expand_dims(y / (1+y) ,1)
    #
    # y = np.concatenate((prob_0,prob_1), axis = 1)
    # print(y[1:10])

    y = np.exp(X[:,0]*X[:,1])

    prob_1 = np.expand_dims(1 / (1+y) ,1)
    prob_0 = np.expand_dims(y / (1+y) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0)

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_additive_labels(X):
    logit = np.exp(-100 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4)

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y



def generate_data(n=100, datatype='', seed = 0, val = False):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples
        datatype(string): The type of data
        choices: 'orange_skin', 'XOR', 'regression'.
        seed: random seed used
    Return:
        X(float): [n,d].
        y(float): n dimensional array.
    """

    np.random.seed(seed)

    X = np.random.randn(n, 10)

    datatypes = None

    if datatype == 'orange_skin':
        y = generate_orange_labels(X)

    elif datatype == 'XOR':
        y = generate_XOR_labels(X)

    elif datatype == 'nonlinear_additive':
        y = generate_additive_labels(X)

    elif datatype == 'alternating':

        # Construct X as a mixture of two Gaussians.
        X[:n//2,-1] += 3
        X[n//2:,-1] += -3
        X1 = X[:n//2]; X2 = X[n//2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:,4:8],X2[:,:4] = X2[:,:4],X2[:,4:8]

        X = np.concatenate([X1,X2], axis = 0)
        y = np.concatenate([y1,y2], axis = 0)

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2))

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]

    elif datatype == 'alternating_xor_orange':

        # Construct X as a mixture of two Gaussians.
        X[:n // 2, -1] += 3
        X[n // 2:, -1] += -3
        X1 = X[:n // 2];
        X2 = X[n // 2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:, 4:8], X2[:, :4] = X2[:, :4], X2[:, 4:8]

        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2))

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]
        datatypes = datatypes[perm_inds]

    elif datatype == 'alternating_xor_nonlin':

        # Construct X as a mixture of two Gaussians.
        X[:n // 2, -1] += 3
        X[n // 2:, -1] += -3
        X1 = X[:n // 2];
        X2 = X[n // 2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:, 4:8], X2[:, :4] = X2[:, :4], X2[:, 4:8]

        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2))

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]
        datatypes = datatypes[perm_inds]



    return X, y, datatypes


def generate_ground_truth(x, data_type):
    """Generate ground truth feature importance corresponding to the data type
       and feature.

    Args:
      - x: features
      - data_type: synthetic data type (syn1 to syn6)
    Returns:
      - ground_truth: corresponding ground truth feature importance
    """

    # Number of samples and features
    n, d = x.shape

    # Output initialization
    ground_truth = np.zeros([n, d])

    # For each data_type
    if data_type == 'syn1':
        ground_truth[:, :2] = 1
    elif data_type == 'syn2':
        ground_truth[:, 2:6] = 1
    elif data_type == 'syn3':
        ground_truth[:, 6:10] = 1

    # Index for syn4, syn5 and syn6
    if data_type in ['syn4', 'syn5', 'syn6']:
        idx1 = np.where(x[:, 10] < 0)[0]
        idx2 = np.where(x[:, 10] >= 0)[0]
        ground_truth[:, 10] = 1

    if data_type == 'syn4':
        ground_truth[idx1, :2] = 1
        ground_truth[idx2, 2:6] = 1
    elif data_type == 'syn5':
        ground_truth[idx1, :2] = 1
        ground_truth[idx2, 6:10] = 1
    elif data_type == 'syn6':
        ground_truth[idx1, 2:6] = 1
        ground_truth[idx2, 6:10] = 1

    return ground_truth


def generate_invase(n=100, data_type='', seed = 0):


    np.random.seed(seed)

    x = np.random.randn(n, 11)

    n = x.shape[0]

    ground_truth = generate_ground_truth(x, data_type)

    datatypes = ground_truth


    # Logit computation
    #logit (not entirely good name) is just the value we obtain from using value v from features and exponentiate it np.exp(v)

    if data_type == 'syn1':
        logit = np.exp(x[:, 0] * x[:, 1])
    elif data_type == 'syn2':
        logit = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
    elif data_type == 'syn3':
        logit = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                       x[:, 8] + np.exp(-x[:, 9]) - 2.4)
    elif data_type == 'syn4':
        logit1 = np.exp(x[:, 0] * x[:, 1])
        logit2 = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
    elif data_type == 'syn5':
        logit1 = np.exp(x[:, 0] * x[:, 1])
        logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                        x[:, 8] + np.exp(-x[:, 9]) - 2.4)
    elif data_type == 'syn6':
        logit1 = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
        logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                        x[:, 8] + np.exp(-x[:, 9]) - 2.4)

        # For syn4, syn5 and syn6 only
        # the output depends only on either first or second dataset (two chosen from syn1,syn2, syn3)
    if data_type in ['syn4', 'syn5', 'syn6']:
        # Based on X[:,10], combine two logits
        #idx1 (2) indiactes if feature 11 is negative (positive)
        idx1 = (x[:, 10] < 0) * 1
        idx2 = (x[:, 10] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2 #sum of two dot products, but for each entry only idx1 or odx2 is 1 so logit is either logit1 or logit2

        # Compute P(Y=0|X)
    prob_0 = np.reshape((logit / (1 + logit)), [n, 1])

    # Sampling process
    y = np.zeros([n, 2])
    y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 1] = 1 - y[:, 0]

    return x, y, datatypes

    # For each data_type

#
# if data_type == 'syn1':
#     ground_truth[:, :2] = 1
# elif data_type == 'syn2':
#     ground_truth[:, 2:6] = 1
# elif data_type == 'syn3':
#     ground_truth[:, 6:10] = 1
#
#     # Index for syn4, syn5 and syn6
# if data_type in ['syn4', 'syn5', 'syn6']:
#     idx1 = np.where(x[:, 10] < 0)[0]
#     idx2 = np.where(x[:, 10] >= 0)[0]
#     ground_truth[:, 10] = 1
#
# if data_type == 'syn4':
#     ground_truth[idx1, :2] = 1
#     ground_truth[idx2, 2:6] = 1
# elif data_type == 'syn5':
#     ground_truth[idx1, :2] = 1
#     ground_truth[idx2, 6:10] = 1
# elif data_type == 'syn6':
#     ground_truth[idx1, 2:6] = 1
#     ground_truth[idx2, 6:10] = 1
