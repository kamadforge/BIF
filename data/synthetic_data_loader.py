

from data.tab_dataloader import load_cervical, load_adult, load_credit
import numpy as np
import os
from pathlib import Path
import pickle
import sys


def synthetic_data_loader(dataset):

    pathmain = Path(__file__).parent.parent

    if dataset == "cervical":

        X_train, y_train, X_test, y_test = load_cervical()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])

    elif dataset == "credit":

        X_train, y_train, X_test, y_test = load_credit()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])

    elif dataset == "adult":

        cwd = os.getcwd()
        pathmain = cwd+"/data/adult/"

        filename = 'adult.p'
        with open(pathmain+filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            y_tot, x_tot = data

    elif dataset == "xor":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/XOR/dataset_XOR.npy'), allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "orange_skin":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/orange_skin/dataset_orange_skin.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "nonlinear_additive":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/nonlinear_additive/dataset_nonlinear_additive.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']


    return x_tot, y_tot