import shap
import os
import xgboost
from pathlib import Path
import os
import pickle
from sklearn import svm
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).resolve().parent.parent / "data"))
sys.path.append(str(Path(sys.path[0]).resolve().parent.parent / "code"))

from publish_bif_tab.code.evaluation_metrics import compute_median_rank, binary_classification_metrics
from publish_bif_tab.data.make_synthetic_datasets import create_data

from publish_bif_tab.data.tab_dataloader import load_adult_short, load_credit, load_intrusion
from publish_bif_tab.data.synthetic_data_loader import synthetic_data_loader
import numpy as np
dataset="subtract" #xor, subtract, nonlinear_additive, orange_skin
dataset_method = f"load_{dataset}"


if "syn" in dataset or dataset=="xor" or "nonlinear" in dataset or "orange" in dataset or "subtract" in dataset:
#     x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset)
#     N_tot, d = x_tot.shape
#     training_data_por = 0.8
#     N = int(training_data_por * N_tot)
#     # if dataset == "adult_short":
#     #     N = 26048
#     # elif dataset == "credit":
#     #     N = 2668
#     X = x_tot[:N, :] #train X
#     y = y_tot[:N] #train y
#     datatypes = datatypes_tot[:N]
#
#     X_test = x_tot[N:, :]
#     y_test = y_tot[N:]
#     datatypes_test = datatypes_tot[N:]
# # elif dataset=="xor" or "nonlinear" in dataset or "orange" in dataset:
# #     X_train, y_train, X_test, y_test = synthetic_data_loader(dataset)
# #     X = X_train
# #     y = y_train
# #     N_tot, d = X_train.shape
# #     N = len(X_train)
# #     datatypes = None

    x_train, y_train, x_test, y_test, datatypes_test, input_shape = create_data(dataset, n=int(1e6))

    N, d = x_train.shape
else:
    X, y, X_test, y_test = globals()[dataset_method]()

input_dim = d
hidden_dim = input_dim
how_many_samps = N

X = x_train
y = y_train
X_test = x_test




model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)

#no absolut value
shap_local_arg = np.argsort(shap_vals)#[::-1]

#absolut value
shap_local_abs = np.abs(shap_vals)
shap_local_abs_arg = np.argsort(-np.abs(shap_vals)) #minus for the reverse order

print(shap_local_abs_arg[0:5])

if dataset=="adult_short":
    dataset="adult"
np.save(f"ranks/shap_{dataset}", shap_local_arg)

#mean_sv = np.abs(shap_values).mean(axis=0)
mean_sv_arg=np.argsort(shap_vals)[::-1]

#np.argsort(np.sum(np.abs(shap_values), axis=0))

#print(mean_sv)
#print(mean_sv_arg)
#print(','.join([str(elem) for elem in mean_sv_arg]) )

if dataset == "xor" or dataset=="subtract":
    k = 2
elif dataset == "orange_skin" or dataset == "nonlinear_additive":
    k = 4
elif dataset == "alternating":
    k = 5
elif dataset == "syn4":
    k = 7
elif dataset == "syn5" or dataset == "syn6":
    k = 9


print(datatypes_test[0:5])

tpr, fdr, mcc = binary_classification_metrics(shap_vals, k, dataset, 2000, datatypes_test, True, shap_local_abs_arg)



print(f"tpr: {tpr}, fdr: {fdr}")
print(f"mcc: {mcc}")

#shap.summary_plot(shap_values, X, plot_type="bar")


######3

#print(shap_values)

#credit qfit
#13  9 11  3 10 16 12  7 15  1  5 21  6 18 19 17 20  2  8 14 25 22 23 24 27  0  4 26 28]
#credit shap
#13 16 3 6 11 6 7 27 25

#

