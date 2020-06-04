import shap
import xgboost
from data.make_synthetic_datasets import generate_data
import numpy as np
import pickle


max_seed = 1
input_dim = 14

# relationship (7), education_num (4), capital_gain (10), race(8), age(0), hours_per_week(12/10),
# [ 7  4  8  0 10  6  9  1  2  5  3 11] # for 12-feature dataset for fairness
#  [ 7  4 10  0 12  6 11  9  5  1  2  8 13  3] for 14-feature dataset for privacy

sv = np.zeros((max_seed,input_dim))

for seed_idx in range(max_seed):

    np.random.seed(seed_idx)

    """ load data """
    filename = 'adult.p'
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, input_dim = x_tot.shape

    # train and test data
    N = np.int(N_tot*0.9)
    rand_perm_nums = np.random.permutation(N_tot)
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]

    # """ load data """
    # X = np.load('X_adult_for_fairness.npy')
    # y = np.load('y_adult_for_fairness.npy')

    N, input_dim = X.shape


    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 500)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)


    mean_sv = np.abs(shap_values).mean(axis=0)
    mean_sv_arg=np.argsort(mean_sv)[::-1]

    print('mean_sv for adult: ', mean_sv)
    print('order of sv for adult: ', mean_sv_arg)
    sv[seed_idx,:] = mean_sv

    print(sv)

# relationship, education num, capital gain [7, 4, 10]
# print(mean_sv)
# print(mean_sv_arg)


# filename = dataset+'shap.npy'
# np.save(filename, sv)
