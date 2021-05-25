from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime.lime_tabular


#np.random.seed(1)

import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).resolve().parent.parent / "data"))
sys.path.append(str(Path(sys.path[0]).resolve().parent.parent / "code"))
import xgboost
from evaluation_metrics import compute_median_rank, binary_classification_metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="nonlinear_additive")
args = parser.parse_args()

from tab_dataloader import load_adult_short, load_credit, load_cervical, load_isolet, load_intrusion
from synthetic_data_loader import synthetic_data_loader
import numpy as np
dataset=args.dataset #nonlinear_additive, orange_skin, xor
dataset_method = f"load_{dataset}"
train_mode=0
test_mode=1
print(dataset)


if "syn" in dataset or dataset=="xor" or "nonlinear" in dataset or "orange" in dataset:
    x_tot, y_tot, datatypes_tot = synthetic_data_loader(dataset)
    N_tot, d = x_tot.shape
    training_data_por = 0.8
    N = int(training_data_por * N_tot)
    # if dataset == "adult_short":
    #     N = 26048
    # elif dataset == "credit":
    #     N = 2668
    X = x_tot[:N, :] #train X
    y = y_tot[:N] #train y
    datatypes = datatypes_tot[:N]  # only for alternating, if datatype comes from orange_skin or
    X_test = x_tot[N:, :]
    y_test = y_tot[N:]
    datatypes_test = datatypes_tot[N:]
# elif dataset=="xor" or "nonlinear" in dataset or "orange" in dataset:
#     X_train, y_train, X_test, y_test = synthetic_data_loader(dataset)
#     X = X_train
#     y = y_train
#     N_tot, d = X_train.shape
#     N = len(X_train)
#     datatypes = None
else:
    X, y, X_test, y_test = globals()[dataset_method]()

input_dim = X.shape[1]
hidden_dim = input_dim
    #how_many_samps = N

if train_mode:
    classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=500) #500
    classifier.fit(X, y)

    # classifier = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    # classifier.fit(X, y)

    #predict_fn = lambda x: rf.predict_proba(X_test)
    score = sklearn.metrics.accuracy_score(y_test, classifier.predict(X_test))
    print("Score: ", score)



    explainer = lime.lime_tabular.LimeTabularExplainer(X, kernel_width=3)

    weight_sum=np.zeros(X_test.shape[1])
    weights_all_local = []
    argsorted_all_local = []

    print(f"Length: {len(X_test)}")
    for i in range(len(X_test)):
        print(i)
        exp = explainer.explain_instance(data_row=X_test[i],predict_fn=classifier.predict_proba, num_features=input_dim)
        exp_list = exp.as_list()
        exp_map = exp.as_map()
        #print(exp_list)

        weights = np.abs(np.array(exp_map[1]))
        weights_ordered = weights[weights[:, 0].argsort()][:,1] #sum of weights from 0 to last feature
        weights_all_local.append(weights_ordered)
        print("weights_argsorted: ", weights_ordered)

        weights_argsorted = np.argsort(weights_ordered)[::-1] #first is most imprortant
        argsorted_all_local.append(weights_argsorted)
        print("weights_argsorted: ", weights_argsorted)

        # just for online output
        weight_sum+=weights_ordered
        weights_sum_argsorted = np.argsort(weight_sum)
        print(f"argsort of sum after {i} iters: {weights_sum_argsorted}")


        #print(m)
        #print(m_sorted)
        #print(weight_sum)
    print(weight_sum)
    sorted_features = np.sort(weight_sum)[::-1]
    argsorted_features = np.argsort(weight_sum)[::-1]
    print("sorted feat:", sorted_features)
    print("argsorted_feat:", argsorted_features)

    print(argsorted_all_local)
    np.save(f"ranks/{dataset}_local", weights_all_local)
    np.save(f"ranks/{dataset}_local_ranks", argsorted_all_local)

if test_mode:
    print("Test")
    file_args = f"ranks/{dataset}_local_ranks.npy"
    file = f"ranks/{dataset}_local.npy"

    ranks = np.load(file_args)
    #ranks = np.flip(ranks)
    vals = np.load(file)
    #vals = np.flip(vals)
    print("loaded:")
    print(ranks[0:5])
    print(vals[0:5])






    if dataset == "xor":
        k = 2
    elif dataset == "orange_skin" or dataset == "nonlinear_additive":
        k = 4
    elif dataset == "alternating":
        k = 5
    elif dataset == "syn4":
        k = 7
    elif dataset == "syn5" or dataset == "syn6":
        k = 9

    print(datatypes_test)

    tpr, fdr, mcc = binary_classification_metrics(vals, k, dataset, len(X_test), datatypes_test, True)

    print(f"tpr: {tpr}, fdr: {fdr}")
    print(f"mcc: {mcc}")


#weights = np.array(exp_list)[:, -1]
#print(weights)



#exp.show_in_notebook(show_table=True)
