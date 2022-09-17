"""
Define functions to evaluate the feature importance under DP and non-DP models
"""

import numpy as np
from scipy.special import digamma
from scipy.special import loggamma
# as digamma


def L2dist(alpha_1, alpha_2):
    # alpha_1 is a numpy array
    # alpha_2 is a numpy array
    alpha_1 = np.array(alpha_1)
    alpha_2 = np.array(alpha_2)
    return np.linalg.norm(alpha_1-alpha_2)

def expected_suff_stats(alpha):
    # alpha is a numpy array
    # eq.(8) of https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    sum_alpha = np.sum(alpha)
    return digamma(alpha) - digamma(sum_alpha)

def KL_Dir(alpha, beta):
    # formula is from
    # http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    alpha_0 = np.sum(alpha)
    beta_0 = np.sum(beta)

    trm1 = loggamma(alpha_0) - loggamma(beta_0)
    trm2 = - np.sum(loggamma(alpha)) + np.sum(loggamma(beta))
    trm3 = np.sum((alpha-beta)*(digamma(alpha)-digamma(alpha_0)))
    KLD = trm1 + trm2 + trm3

    return KLD

def KL_Bern(p, q):
    # formula is from
    # https://math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution

    p = np.array(p)
    q = np.array(q)
    trm1 = p*np.log(p/q)
    trm2 = (1-p)*np.log((1-p)/(1-q))
    KLD = np.sum(trm1 + trm2)

    return KLD





