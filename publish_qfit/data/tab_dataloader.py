import socket
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import math

if sys.version_info[0] > 2:
    import sdgym
# import xgboost
import pickle
from pathlib import Path
import os

# cleaned for publishing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
#import xgboost

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
#from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score


#################################


def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected





def load_credit():

    seed_number=0
    np.random.seed(seed_number)


    print("Creditcard fraud detection dataset") # this is homogeneous


    credit_path=""

    if len(credit_path)==0:
        print("Please input the creditcard fraud dataset path")
        exit()

    data = pd.read_csv(credit_path)
        # data = pd.read_csv(
        #    "../data/Kaggle_Credit/creditcard.csv")



    feature_names = data.iloc[:, 1:30].columns
    target = data.iloc[:1, 30:].columns

    data_features = data[feature_names]
    data_target = data[target]
    print(data_features.shape)

    """ we take a pre-processing step such that the dataset is a bit more balanced """
    raw_input_features = data_features.values
    raw_labels = data_target.values.ravel()

    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = 0.01# undersampled_rate #0.01
    # under_sampling_rate = 0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                        test_size=0.20, random_state=seed_number)
    n_classes = 2

    return X_train, y_train, X_test, y_test.squeeze()




def load_adult():
    seed_number=0
    print("dataset is adult") # this is heterogenous
    print(socket.gethostname())
    #if 'g0' not in socket.gethostname():
    data, categorical_columns, ordinal_columns = sdgym.load_dataset('adult')
    # else:

    """ some specifics on this dataset """
    numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
    n_classes = 2

    data = data[:, numerical_columns + ordinal_columns + categorical_columns]

    num_numerical_inputs = len(numerical_columns)
    num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

    inputs = data[:, :-1]
    target = data[:, -1]

    inputs, target=undersample(inputs, target, 0.4)

    X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                        random_state=seed_number)

    return X_train, y_train, X_test, y_test

def load_adult_short():

    seed_number=0
    np.random.seed(seed_number)

    #if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():

    cwd = Path(__file__).parent.parent
    pathmain = os.path.join(cwd, "data/adult/")

    filename = 'adult.p'
    with open(os.path.join(pathmain, filename), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
        y_tot, x_tot = data

    X_train, X_test, y_train, y_test = train_test_split(x_tot, y_tot, train_size=0.80, test_size=0.20,
                                                        random_state=0)

    return X_train, y_train, X_test, y_test



