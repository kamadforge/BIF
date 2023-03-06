import sys
import os
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp import rdp_bank
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism,SubsampleGaussianMechanism, GaussianMechanism, ComposedGaussianMechanism, LaplaceMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling
# matplotlib inline


# (2) desired delta level
delta = 1e-5
eps = 0.0001

# (5) number of training steps
n_epochs = 20  # for training the classifier
batch_size = 1000  # the same across experiments

n_data = 18089 # this is for diabetic dataset
steps_per_epoch = n_data // batch_size
n_steps = steps_per_epoch * n_epochs
# n_steps = 1

# (6) sampling rate
prob = batch_size / n_data


general_calibrate = generalized_eps_delta_calibrator()
params = {}
coeff = 20
params['sigma'] = None
params['prob'] = prob
params['coeff'] = coeff
mech4 = general_calibrate(SubsampleGaussianMechanism, eps, delta, [0,1000],params=params,para_name='sigma', name='Subsampled_Gaussian')
print(mech4.name, mech4.params, mech4.get_approxDP(delta))


# sigma = 504 for eps = 0.001
# sigma = 68.7 for eps = 0.01
# sigma = 16.3 for eps = 0.05
# sigma = 8.8 for eps = 0.1
# sigma = 1.6 for eps = 0.5
# sigma = 2.4 for eps = 1.0
# sigma = 1.14 for eps = 2.0
# sigma = 0.84 for eps = 4.0
# sigma = 0.64 for eps = 8.0
# sigma = 0.61 for eps = 10.0