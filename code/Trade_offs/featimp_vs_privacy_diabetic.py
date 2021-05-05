"""
Test learning feature importance under DP and non-DP models
"""
__author__ = 'anon_m'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import argparse
from torch.distributions import Gamma
import pickle
import numpy.random as rn
from Models import Feedforward, Feature_Importance_Model
from sklearn.metrics import roc_auc_score
from Losses import loss_function

from dp_sgd import dp_sgd_backward
from backpack import extend
import shap
import xgboost
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib

def main():

    ar = parse()
    rn.seed(ar.seed)

    """ load data """
    X = np.load('X_tr_diabetic.npy', allow_pickle=True)
    y = np.load('y_tr_diabetic.npy', allow_pickle=True)

    Xtst = np.load('X_tst_diabetic.npy', allow_pickle=True)
    ytst = np.load('y_tst_diabetic.npy', allow_pickle=True)

    column_names = np.load('column_names_diabetic.npy', allow_pickle=True)

    use_cuda = not ar.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    """ set the privacy parameter """
    # dp_epsilon = 1
    # dp_delta = 1/N_tot
    # k = ? , # state the number of iterations
    # params = privacy_calibrator.gaussian_mech(dp_epsilon, dp_delta, prob=nu, k=k)
    # sigma = params['sigma']
    # print('privacy parameter is ', sigma)

    # train and test data
    N, input_dim = X.shape
    print('n_data:', N)

#############################################################
    """ train a classifier """
#############################################################
    classifier = Feedforward(input_dim, 200, 50).to(device)
    if ar.load_saved_model:
        loaded_states = torch.load(f'dp_classifier_sig{ar.dp_sigma}.pt')
        clean_states = dict()
        for key in classifier.state_dict().keys():
            clean_states[key] = loaded_states[key]
        classifier.load_state_dict(clean_states)
        classifier.eval()
    else:
        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(classifier.parameters(), lr=ar.lr)

        classifier.train()
        if ar.dp_sigma > 0.:
            extend(classifier)
        # how_many_epochs = 10
        # mini_batch_size = 100
        how_many_iter = np.int(N / ar.clf_batch_size)

        for epoch in range(ar.clf_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):
                # get the inputs
                inputs = torch.Tensor(X[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size, :]).to(device)
                labels = torch.Tensor(y[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size]).to(device)

                optimizer.zero_grad()
                y_pred = classifier(inputs)
                loss = criterion(y_pred.squeeze(), labels)

                if ar.dp_sigma > 0.:
                    global_norms, global_clips = dp_sgd_backward(classifier.parameters(), loss, device, ar.dp_clip, ar.dp_sigma)
                    # print(f'max_norm:{torch.max(global_norms).item()}, mean_norm:{torch.mean(global_norms).item()}')
                    # print(f'mean_clip:{torch.mean(global_clips).item()}')
                else:
                    loss.backward()
                optimizer.step()
                running_loss += loss.item()

                y_pred = classifier(torch.Tensor(Xtst).to(device))
                ROC = roc_auc_score(ytst, y_pred.cpu().detach().numpy())
                PRC = average_precision_score(ytst, y_pred.cpu().detach().numpy())
                print('Epoch {}: ROC : {}'.format(epoch, ROC))
                print('Epoch {}: PRC : {}'.format(epoch, PRC))

        print('Finished Classifier Training')

        if ar.save_model:
            print('saving model')
            torch.save(classifier.state_dict(), f'dp_classifier_sig{ar.dp_sigma}.pt')
            filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'roc.npy'
            np.save(filename, ROC)



#############################################################
    """ learn feature importance """
#############################################################

    # we initialize the parameters based on SHAP estimation.
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_sv = np.abs(shap_values).mean(axis=0)

    order_by_importance_shap = np.argsort(mean_sv)[::-1] # descending order
    print('importance by SHAP: ', column_names[order_by_importance_shap])


    # init_phi = 100 * mean_sv
    init_phi = 5*np.ones(mean_sv.shape)
    # print('phi init: ', init_phi)

    seedmax = 1

    mean_importance = np.zeros((seedmax, input_dim))
    phi_est_mat = np.zeros((seedmax, input_dim))

    for seednum in range(0,seedmax):

        rn.seed(seednum)

        """ learn feature importance """
        num_Dir_samps = 100
        importance = Feature_Importance_Model(input_dim, classifier, num_Dir_samps, init_phi, device).to(device)
        # optimizer = optim.SGD(importance.parameters(), lr=0.01)
        optimizer = optim.Adam(importance.parameters(), lr=0.001)

        # We freeze the classifier
        ct = 0
        for child in importance.children():
            ct +=1
            if ct>=1:
                for param in child.parameters():
                    param.requires_grad = False

        # print(list(importance.parameters())) # make sure I only update the gradients of feature importance

        importance.train()
        how_many_iter = np.int(N / ar.switch_batch_size)

        alpha_0 = 0.5
        annealing_rate = 1  # we don't anneal. don't want to think about this.
        kl_term = True

        for epoch in range(ar.switch_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                inputs = torch.Tensor(X[i * ar.switch_batch_size:(i + 1) * ar.switch_batch_size, :]).to(device)
                labels = torch.Tensor(y[i * ar.switch_batch_size:(i + 1) * ar.switch_batch_size]).to(device)

                optimizer.zero_grad()
                y_pred, phi_cand = importance(inputs)
                labels = torch.squeeze(labels)
                loss = loss_function(y_pred, labels.view(-1, 1).repeat(1, num_Dir_samps), phi_cand, alpha_0, input_dim,
                                     annealing_rate, N, kl_term, device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if np.remainder(epoch,50)==0:
                print('Epoch {}: running_loss : {}'.format(epoch, running_loss/how_many_iter))
                estimated_params = list(importance.parameters())
                phi_est = F.softplus(estimated_params[0].cpu())
                print('phi_est', phi_est.detach().numpy())

        print('Finished feature importance Training')

        """ checking the results """
        estimated_params = list(importance.parameters())
        phi_est = F.softplus(estimated_params[0].cpu())

        concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
        beta_param = torch.ones(concentration_param.size())
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample()
        Sstack = gamma_samps / torch.sum(gamma_samps, 0)
        avg_S = torch.mean(Sstack, 1)
        posterior_mean_switch = avg_S.detach().numpy()
        print('estimated posterior mean of feature importance is', posterior_mean_switch)

        mean_importance[seednum,:] = posterior_mean_switch
        phi_est_mat[seednum,:] = phi_est.detach().numpy()

        order_by_importance = np.argsort(posterior_mean_switch)[::-1] # descending order
        # print('seed number: ', seednum)
        # print('order by importance: ', order_by_importance)
        print('importance by Switches: ', column_names[order_by_importance])

    # store the result
    filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'importance.npy'
    np.save(filename, mean_importance)

    filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'phi_est.npy'
    np.save(filename, phi_est_mat)

    # filename = 'pri_' + str(ar.dp_sigma) + 'seed_' + str(ar.seed) + 'roc.npy'
    # np.save(filename, ROC)


    ## plotting the results
    font = {
        # 'family': 'normal',
            # 'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    # fig, axs = plt.subplots(1, 2, figsize = (20, 10))
    top_few = [0,1,2,3,4,5,6]

    plt.figure(1)
    sns.barplot(y = [element for element in column_names[order_by_importance_shap][top_few]],
                 x = [element for element in mean_sv[order_by_importance_shap][top_few]])
    plt.title('SHAP (non_priv)')

    plt.figure(2)
    sns.barplot(y = [element for element in column_names[order_by_importance][top_few]],
                x = [element for element in posterior_mean_switch[order_by_importance][top_few]])
    if ar.dp_sigma==1e-6:
        plt.title("BIF (non_priv)")
    else:
        plt.title("BIF (sigma=%.2f)" %ar.dp_sigma)
    plt.show()


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clf-epochs', type=int, default=20)
    parser.add_argument('--clf-batch-size', type=int, default=1000)

    parser.add_argument('--switch-epochs', type=int, default=250)
    parser.add_argument('--switch-batch-size', type=int, default=500)
    parser.add_argument('--save-model', action='store_true', default=True)
    parser.add_argument('--load-saved-model', action='store_true', default=True)
    # parser.add_argument('--switch-batch-size', type=int, default=20000)

    parser.add_argument('--dp-clip', type=float, default=0.01)


    # sigma = 68.7 for eps = 0.01
    # sigma = 8.8 for eps = 0.1
    # sigma = 2.4 for eps = 1.0
    # sigma = 0.84 for eps = 4.0
    # sigma = 1e-6 for eps = infty (nonprivate)

    parser.add_argument('--dp-sigma', type=float, default=8.8)
    # parser.add_argument('--dp-clip', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--dp-sigma', type=float, default=1.35)
    # parser.add_argument('--dp-sigma', type=float, default=2.3)
    # parser.add_argument('--dp-sigma', type=float, default=4.4)
    # parser.add_argument('--dp-sigma', type=float, default=8.4)
    # parser.add_argument('--dp-sigma', type=float, default=17.)
    return parser.parse_args()


if __name__ == '__main__':
    main()
