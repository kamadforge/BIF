import shap
import xgboost
from data.make_synthetic_datasets import generate_data
import numpy as np


max_seed = 5
input_dim = 10
sv = np.zeros((max_seed,input_dim))

for seed_idx in range(max_seed):

    np.random.seed(seed_idx)

    """ generate data """
    N_tot = 10000
    dataset = 'XOR'
    # dataset = 'orange_skin'
    # dataset = 'nonlinear_additive'
    x_tot, y_tot, datatypes = generate_data(N_tot, dataset)
    y_tot = np.argmax(y_tot, axis=1)


    # train and test data
    N = np.int(N_tot*0.9)
    rand_perm_nums = np.random.permutation(N_tot)
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]


    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)


    mean_sv = np.abs(shap_values).mean(axis=0)
    # mean_sv_arg=np.argsort(mean_sv)[::-1]
    sv[seed_idx,:] = mean_sv


# print(mean_sv)
# print(mean_sv_arg)


filename = dataset+'shap.npy'
np.save(filename, sv)
