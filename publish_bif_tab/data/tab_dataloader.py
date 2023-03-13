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



def load_intrusion():

    seed_number=0

    print("dataset is intrusion")
    print(socket.gethostname())
    data, categorical_columns, ordinal_columns = sdgym.load_dataset('intrusion')

    """ some specifics on this dataset """
    n_classes = 5 #removed to 5

    """ some changes we make in the type of features for applying to our model """
    categorical_columns_binary = [6, 11, 13, 20]  # these are binary categorical columns
    the_rest_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns_binary))

    num_numerical_inputs = len(the_rest_columns)  # 10. Separately from the numerical ones, we compute the length-scale for the rest columns
    num_categorical_inputs = len(categorical_columns_binary)  # 4.

    raw_labels = data[:, -1]
    raw_input_features = data[:, the_rest_columns + categorical_columns_binary]
    print(raw_input_features.shape)

    #we remove the least label
    non4_tokeep=np.where(raw_labels!=4)[0]
    raw_labels=raw_labels[non4_tokeep]
    raw_input_features=raw_input_features[non4_tokeep]

    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0  # this is a dominant one about 80%, which we want to undersample
    idx_positive_label = raw_labels != 0

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 40% of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = 0.3#undersampled_rate#0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80, test_size=0.20, random_state=seed_number)

    return X_train, y_train, X_test, y_test




def load_covtype():

    seed_number=0

    print("dataset is covtype")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
        train_data = np.load("/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/data/real/covtype/train.npy")
        test_data = np.load("/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/data/real/covtype/test.npy")
        # we put them together and make a new train/test split in the following
        data = np.concatenate((train_data, test_data))
    else:
        train_data = np.load(
            "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/train.npy")
        test_data = np.load(
            "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/test.npy")
        data = np.concatenate((train_data, test_data))

    """ some specifics on this dataset """
    numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ordinal_columns = []
    categorical_columns = list(set(np.arange(data.shape[1])) - set(numerical_columns + ordinal_columns))
    # Note: in this dataset, the categorical variables are all binary
    n_classes = 7

    print('data shape is', data.shape)
    print('indices for numerical columns are', numerical_columns)
    print('indices for categorical columns are', categorical_columns)
    print('indices for ordinal columns are', ordinal_columns)

    # sorting the data based on the type of features.
    data = data[:, numerical_columns + ordinal_columns + categorical_columns]
    # data = data[0:150000, numerical_columns + ordinal_columns + categorical_columns] # for fast testing the results

    num_numerical_inputs = len(numerical_columns)
    num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

    inputs = data[:20000, :-1]
    target = data[:20000, -1]


    ##################3

    raw_labels=target
    raw_input_features=inputs

    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 1  # 1 and 0 are dominant but 1 has more labels
    idx_positive_label = raw_labels != 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 40% of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = 0.3#undersampled_rate  # 0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))


    ###############3

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                        random_state=seed_number)  # 60% training and 40% test

    return X_train, y_train, X_test, y_test




def load_credit():

    seed_number=0
    np.random.seed(seed_number)


    print("Creditcard fraud detection dataset") # this is homogeneous

    # if len(credit_path)==0:
    #     print("Please input the creditcard fraud dataset path")
    #     exit()

    data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")

    # if 'kamil' in socket.gethostname():
    #
    #     # data = pd.read_csv(
    #     #     "/home/kamil/Dropbox/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv")
    #
    #
    # # if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
    # else:
    #     # (1) load data
    #     data = pd.read_csv(
    #         '/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv')


    #
    # data = pd.read_csv(credit_path)
    #     # data = pd.read_csv(
    #     #    "../data/Kaggle_Credit/creditcard.csv")
    #


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



