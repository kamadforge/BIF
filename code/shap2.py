import shap

import xgboost
from pathlib import Path
import os
import pickle
from sklearn import svm

from data.tab_dataloader import load_adult_short, load_credit, load_cervical
import numpy as np


X, y, X_test, y_test = load_cervical()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


mean_sv = np.abs(shap_values).mean(axis=0)
mean_sv_arg=np.argsort(mean_sv)[::-1]

#np.argsort(np.sum(np.abs(shap_values), axis=0))

print(mean_sv)
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
