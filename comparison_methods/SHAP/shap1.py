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

from tab_dataloader import load_adult_short, load_credit, load_cervical, load_isolet, load_intrusion
import numpy as np
dataset="intrusion"
dataset_method = f"load_{dataset}"

X, y, X_test, y_test = globals()[dataset_method]()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)

#no absolut value
shap_local_arg = np.argsort(shap_vals)[::-1]

#absolut value
shap_local_abs = np.abs(shap_vals)
shap_local_abs_arg = np.argsort(np.abs(shap_vals))[::-1]

np.save(f"ranks/shap_{dataset}", shap_local_abs_arg)

#mean_sv = np.abs(shap_values).mean(axis=0)
mean_sv_arg=np.argsort(shap_vals)[::-1]

#np.argsort(np.sum(np.abs(shap_values), axis=0))

#print(mean_sv)
print(mean_sv_arg)
print(','.join([str(elem) for elem in mean_sv_arg]) )

#shap.summary_plot(shap_values, X, plot_type="bar")


######3

#print(shap_values)

#credit qfit
#13  9 11  3 10 16 12  7 15  1  5 21  6 18 19 17 20  2  8 14 25 22 23 24 27  0  4 26 28]
#credit shap
#13 16 3 6 11 6 7 27 25

#

