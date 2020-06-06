"""
Define functions to evaluate the feature importance under DP and non-DP models
"""

__author__ = 'anon_m'

import numpy as np
import scipy.special.digamma as digamma


def L2dist(alpha_1, alpha_2):
    # alpha_1 is a numpy array
    # alpha_2 is a numpy array
    return np.norm(alpha_1-alpha_2)

def expected_suff_stats(alpha):
    # alpha is a numpy array
    # eq.(8) of https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    sum_alpha = np.sum(alpha)
    return digamma(alpha) - digamma(sum_alpha)


