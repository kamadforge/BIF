import shap

import xgboost
from pathlib import Path
import os
import pickle
from sklearn import svm

# train XGBoost model
# X,y = shap.datasets.boston()


# X_train, y_train, X_test, y_test = load_credit()
# x_tot = np.concatenate([X_train, X_test])
# y_tot = np.concatenate([y_train, y_test])

cwd = Path(__file__).parent.parent
pathmain = os.path.join(cwd, "data/adult/")

filename = 'adult.p'
with open(os.path.join(pathmain, filename), 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    y_tot, x_tot = data


 # unpack data
    N_tot, d = x_tot.shape

    training_data_por = 0.8

    N = int(training_data_por * N_tot)

    X = x_tot[:N, :]
    y = y_tot[:N]


    X_test = x_tot[N:, :]
    y_test = y_tot[N:]

    input_dim = d
    hidden_dim = input_dim
    how_many_samps = N



svm = svm.SVC(kernel='rbf', probability=True)
svm.fit(X, y)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

print(shap_values)