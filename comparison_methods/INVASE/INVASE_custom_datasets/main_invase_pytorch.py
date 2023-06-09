"""Main function for INVASE.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
           "IINVASE: Instance-wise Variable Selection using Neural Networks,"
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com

---------------------------------------------------

(1) Data generation
(2) Train INVASE or INVASE-
(3) Evaluate INVASE on ground truth feature importance and prediction
"""

# Necessary packages
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import numpy as np
import torch as pt
import sys

sys.path.append("/home/kamil/Dropbox/Current_research/featimp_dp")
from comparison_methods.INVASE.data_generation import generate_dataset
from comparison_methods.INVASE.INVASE_custom_datasets.invase_pytorch import Invase
from comparison_methods.INVASE.utils import feature_performance_metric, prediction_performance_metric

sys.path.append("/publish_bif_tab")
from publish_bif_tab.data.make_synthetic_datasets import generate_data_forinvasecode


# try:
#  a=3
# except ImportError:
#   # noinspection PyUnresolvedReferences
#   from data_generation import generate_dataset
#   # noinspection PyUnresolvedReferences
#   from invase_pytorch import Invase
#   # noinspection PyUnresolvedReferences
#   from utils import feature_performance_metric, prediction_performance_metric

from publish_bif_tab.data.tab_dataloader import load_adult_short, load_credit, load_intrusion



def main(args):
  """Main function for INVASE.

  Args:
    - data_type: synthetic data type (syn1 to syn6)
    - train_no: the number of samples for training set
    - train_no: the number of samples for testing set
    - dim: the number of features
    - model_type: invase or invase_minus
    - model_parameters:
      - actor_h_dim: hidden state dimensions for actor
      - critic_h_dim: hidden state dimensions for critic
      - n_layer: the number of layers
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE

  Returns:
    - performance:
      - mean_tpr: mean value of true positive rate
      - std_tpr: standard deviation of true positive rate
      - mean_fdr: mean value of false discovery rate
      - std_fdr: standard deviation of false discovery rate
      - auc: area under roc curve
      - apr: average precision score
      - acc: accuracy
  """
  print('#################### generating data')
  if "syn" in args.data_type:
  # Generate dataset
    x_train, y_train, g_train = generate_dataset(n=args.train_no, dim=args.dim, data_type=args.data_type, seed=0)

    x_test, y_test, g_test = generate_dataset(n=args.test_no, dim=args.dim, data_type=args.data_type, seed=0)

    x_train, y_train, g_train = generate_data_forinvasecode(10000, args.data_type)
    x_test, y_test, g_test = generate_data_forinvasecode(10000, args.data_type)

    x_train, y_train, g_train = generate_data_forinvasecode(10000, args.data_type)
    x_test, y_test, g_test = generate_data_forinvasecode(10000, args.data_type)
  elif args.data_type == "adult_short":
    x_train, y_train, x_test, y_test = load_adult_short()
  elif args.data_type == "intrusion":
    x_train, y_train, x_test, y_test = load_intrusion()

  model_parameters = {'lamda': args.lamda,
                      'actor_h_dim': args.actor_h_dim,
                      'critic_h_dim': args.critic_h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration,
                      'activation': args.activation,
                      'learning_rate': args.learning_rate}

  device = pt.device("cuda" if not args.no_cuda else "cpu")

  print('#################### training model')
  # Train the model
  model = Invase(x_train, y_train, args.model_type, model_parameters, device)

  model.run_training(x_train, y_train)
  model.train(False)

  print('#################### evaluating')
  # # Evaluation
  # Compute importance score
  g_hat = model.importance_score(x_test)

  feature_mean=g_hat.mean(axis=0)
  features_sorted=np.argsort(feature_mean)[::-1]
  features_sorted_string=",".join([str(a) for a in features_sorted])

  with np.printoptions(precision=3, suppress=True):
    print(feature_mean)
  print(features_sorted_string) #features, averaged over samples

  print()
  importance_score = 1. * (g_hat > 0.5)

  if "syn" in args.data_type:
    # Evaluate the performance of feature importance
    # mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(g_test, importance_score)
    mean_tpr, std_tpr, mean_fdr, std_fdr, mcc = feature_performance_metric(g_test, importance_score)


    # Print the performance of feature importance
    print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '%, ' + 'TPR std: ' + str(np.round(std_tpr, 1)) + '%, ')
    print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '%, ' + 'FDR std: ' + str(np.round(std_fdr, 1)) + '%, ')
    print('MCC: ' + str(mcc))

  # Predict labels
  else:
    #for real orl datasets

    #pruning here
    #ascending
    instance_best_features_ascending = np.argsort(importance_score, axis=1)
    instance_unimportant_features=instance_best_features_ascending[:, :-args.ktop_features]
    np.save(f"ranks/instance_featureranks_test_invase_{args.data_type}_k_{args.ktop_features}_iteration_{args.iteration}.npy", instance_unimportant_features)

    #
    # for i, data in enumerate(x_test):
    #   x_test[i, instance_unimportant_features[i]]=0


    y_hat = model.predict(x_test)

    # uncomment for evaluation
    # Evaluate the performance of feature importance
    auc, apr, acc = prediction_performance_metric(y_test, y_hat)

    # Print the performance of feature importance
    print('AUC: ' + str(np.round(auc, 3)) + ', APR: ' + str(np.round(apr, 3)) + ', ACC: ' + str(np.round(acc, 3)))

    performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr, 'mean_fdr': mean_fdr,
                   'std_fdr': std_fdr, 'auc': auc, 'apr': apr, 'acc': acc}



  #return performance


##
if __name__ == '__main__':
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_type', choices=['syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6'], default='syn4', type=str)
  parser.add_argument('--train_no', help='the number of training data', default=10000, type=int)
  parser.add_argument('--test_no', help='the number of testing data', default=10000, type=int)
  parser.add_argument('--dim', help='the number of features', choices=[11, 100], default=11, type=int)
  parser.add_argument('--lamda', help='inavse hyper-parameter lambda', default=0.1, type=float)
  parser.add_argument('--actor_h_dim', help='hidden state dimensions for actor', default=100, type=int)
  parser.add_argument('--critic_h_dim', help='hidden state dimensions for critic', default=200, type=int)
  parser.add_argument('--n_layer', help='the number of layers', default=3, type=int)
  parser.add_argument('--batch_size', help='the number of samples in mini batch', default=1000, type=int)
  parser.add_argument('--iteration', help='the number of iteration', default=1000, type=int) #10000, 3000
  parser.add_argument('--activation', help='activation function of the networks',
                      choices=['selu', 'relu'], default='relu', type=str)
  parser.add_argument('--learning_rate', help='learning rate of model training', default=0.0001, type=float)
  parser.add_argument('--model_type', help='inavse or invase- (without baseline)',
                      choices=['invase', 'invase_minus'], default='invase_minus', type=str)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument("--ktop_features", default=5)
  args_in = parser.parse_args()

  # Call main function
  # performance = main(args_in)
  main(args_in)
