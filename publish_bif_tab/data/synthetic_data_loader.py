

from tab_dataloader import load_adult, load_credit, load_adult_short, load_intrusion
import numpy as np
import os
from pathlib import Path
import pickle
import sys


def synthetic_data_loader(dataset):

    pathmain = Path(__file__).parent.parent
    datatypes = None



    if dataset == "credit":

        X_train, y_train, X_test, y_test = load_credit()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])


    elif dataset == "adult":

        # X_train, y_train, X_test, y_test = load_credit()
        # x_tot = np.concatenate([X_train, X_test])
        # y_tot = np.concatenate([y_train, y_test])

        cwd = Path(__file__).parent.parent
        pathmain = os.path.join(cwd,"data/adult/")

        filename = 'adult.p'
        with open(os.path.join(pathmain,filename), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            y_tot, x_tot = data

    elif dataset == "adult_short":
        X_train, y_train, X_test, y_test = load_adult_short()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])

    elif dataset == "intrusion":

        X_train, y_train, X_test, y_test = load_intrusion()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])


    elif dataset == "adult_short":
        X_train, y_train, X_test, y_test = load_adult_short()
        x_tot = np.concatenate([X_train, X_test])
        y_tot = np.concatenate([y_train, y_test])

    elif dataset == "xor":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/XOR/dataset_XOR.npy'), allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "subtract":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/subtract/dataset_subtract.npy'), allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "xor_mean5":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/XOR/dataset_XOR_mean5.npy'), allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "orange_skin":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/orange_skin/dataset_orange_skin.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "orange_skin_mean5":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/orange_skin/dataset_orange_skin_mean5.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "nonlinear_additive":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/nonlinear_additive/dataset_nonlinear_additive.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']

    elif dataset == "alternating":


        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/alternating/dataset_alternating.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']

    elif dataset == "syn4":

        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/invase/dataset_syn4.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']

    elif dataset == "syn4_mean5":

        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/invase/dataset_syn4_mean5.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']

    elif dataset == "syn5":

        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/invase/dataset_syn5.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']

    elif dataset == "syn6":

        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/invase/dataset_syn6.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']

    elif dataset =="total":

        xor_dataset = np.load(os.path.join(pathmain, 'data/synthetic/qtip/dataset_total.npy'),
                              allow_pickle=True)
        x_tot = xor_dataset[()]['x']
        y_tot = xor_dataset[()]['y']
        datatypes = xor_dataset[()]['datatypes']




    return x_tot, y_tot, datatypes