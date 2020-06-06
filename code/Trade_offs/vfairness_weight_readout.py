"""
Study trade-off between global feature importance and fairness

"""

__author__ = 'anon_m'

# 29 May 2020
""" Fairness part came from https://github.com/equialgo/fairness-in-ml/blob/master/fairness-in-ml.ipynb """

import pandas as pd
import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")
from IPython import display

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sdgym import load_dataset
import pickle

# def load_ICU_data(path):
#   column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
#                   'marital_status', 'occupation', 'relationship', 'race', 'sex',
#                   'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
#
#   input_data = (pd.read_csv(path, names=column_names,
#                             na_values="?", sep=r'\s*,\s*', engine='python')
#     .loc[lambda df: df['race'].isin(['White', 'Black'])])
#
#   # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
#   sensitive_attribs = ['race', 'sex']
#   Z = (input_data.loc[:, sensitive_attribs]
#        .assign(race=lambda df: (df['race'] == 'White').astype(int),
#                sex=lambda df: (df['sex'] == 'Male').astype(int)))
#
#   # targets; 1 when someone makes over 50k , otherwise 0
#   y = (input_data['target'] == '>50K').astype(int)
#
#   # features; note that the 'target' and sentive attribute columns are dropped
#   X = (input_data
#        .drop(columns=['target', 'race', 'sex'])
#        .fillna('Unknown')
#        .pipe(pd.get_dummies, drop_first=True))
#
#   # to check out the column names
#   # for col in X.columns:
#   #   print(col)
#
#   print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
#   print(f"targets y: {y.shape[0]} samples")
#   print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
#   return X, y, Z


def nn_classifier(n_features):
  inputs = Input(shape=(n_features,))
  dense1 = Dense(32, activation='relu')(inputs)
  dropout1 = Dropout(0.2)(dense1)
  dense2 = Dense(32, activation='relu')(dropout1)
  dropout2 = Dropout(0.2)(dense2)
  dense3 = Dense(32, activation="relu")(dropout2)
  dropout3 = Dropout(0.2)(dense3)
  outputs = Dense(1, activation='sigmoid')(dropout3)
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model


def plot_distributions(y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None):
  fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
  legend={'race': ['black','white'],
          'sex': ['female','male']}
  for idx, attr in enumerate(Z.columns):
    for attr_val in [0, 1]:
      ax = sns.distplot(y[Z[attr] == attr_val], hist=False,
                        kde_kws={'shade': True,},
                        label='{}'.format(legend[attr][attr_val]),
                        ax=axes[idx])
    ax.set_xlim(0,1)
    ax.set_ylim(0,7)
    ax.set_yticks([])
    ax.set_title("sensitive attibute: {}".format(attr))
    if idx == 0:
      ax.set_ylabel('prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(attr))
  if iteration:
    fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize='16')
  if val_metrics is not None:
    fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                   f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                   f"- Accuracy: {val_metrics['Accuracy']:.1f}"]),
             fontsize='16')
  if p_rules is not None:
    fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                 [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                  for attr in p_rules.keys()]),
             fontsize='16')
  fig.tight_layout()
  if fname is not None:
    plt.savefig(fname, bbox_inches='tight')
  return fig


def p_rule(y_pred, z_values, threshold=0.5):
  y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
  y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
  odds = y_z_1.mean() / y_z_0.mean()
  return np.min([odds, 1/odds]) * 100


class FairClassifier(object):

  def __init__(self, n_features, n_sensitive, lambdas):
    self.lambdas = lambdas

    clf_inputs = Input(shape=(n_features,))
    adv_inputs = Input(shape=(1,))

    clf_net = self._create_clf_net(clf_inputs)
    adv_net = self._create_adv_net(adv_inputs, n_sensitive)
    self._trainable_clf_net = self._make_trainable(clf_net)
    self._trainable_adv_net = self._make_trainable(adv_net)
    self._clf = self._compile_clf(clf_net)
    self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
    self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
    self._val_metrics = None
    self._fairness_metrics = None

    self.predict = self._clf.predict

  def _make_trainable(self, net):
    def make_trainable(flag):
      net.trainable = flag
      for layer in net.layers:
        layer.trainable = flag

    return make_trainable

  def _create_clf_net(self, inputs):
    dense1 = Dense(32, activation='relu')(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(32, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    outputs = Dense(1, activation='sigmoid', name='y')(dropout3)
    return Model(inputs=[inputs], outputs=[outputs])

  def _create_adv_net(self, inputs, n_sensitive):
    dense1 = Dense(32, activation='relu')(inputs)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    outputs = [Dense(1, activation='sigmoid')(dense3) for _ in range(n_sensitive)]
    return Model(inputs=[inputs], outputs=outputs)

  def _compile_clf(self, clf_net):
    clf = clf_net
    self._trainable_clf_net(True)
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    return clf

  def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
    clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs)] + adv_net(clf_net(inputs)))
    self._trainable_clf_net(True)
    self._trainable_adv_net(False)
    loss_weights = [1.] + [-lambda_param for lambda_param in self.lambdas]
    clf_w_adv.compile(loss=['binary_crossentropy'] * (len(loss_weights)),
                      loss_weights=loss_weights,
                      optimizer='adam')
    return clf_w_adv

  def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
    adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
    self._trainable_clf_net(False)
    self._trainable_adv_net(True)
    adv.compile(loss=['binary_crossentropy'] * n_sensitive, optimizer='adam')
    return adv

  def _compute_class_weights(self, data_set):
    class_values = [0, 1]
    class_weights = []
    if len(data_set.shape) == 1:
      balanced_weights = compute_class_weight('balanced', class_values, data_set)
      class_weights.append(dict(zip(class_values, balanced_weights)))
    else:
      n_attr = data_set.shape[1]
      for attr_idx in range(n_attr):
        balanced_weights = compute_class_weight('balanced', class_values,
                                                np.array(data_set)[:, attr_idx])
        class_weights.append(dict(zip(class_values, balanced_weights)))
    return class_weights

  def _compute_target_class_weights(self, y):
    class_values = [0, 1]
    balanced_weights = compute_class_weight('balanced', class_values, y)
    class_weights = {'y': dict(zip(class_values, balanced_weights))}
    return class_weights

  def pretrain(self, x, y, z, epochs=10, verbose=0):
    self._trainable_clf_net(True)
    # self._clf.fit(x.values, y.values, epochs=epochs, verbose=verbose)
    self._clf.fit(x, y, epochs=epochs, verbose=verbose)
    self._trainable_clf_net(False)
    self._trainable_adv_net(True)
    class_weight_adv = self._compute_class_weights(z)
    # self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), class_weight=class_weight_adv,
    #               epochs=epochs, verbose=verbose)
    self._adv.fit(x, np.hsplit(z, z.shape[1]), class_weight=class_weight_adv,
                  epochs=epochs, verbose=verbose)

  def fit(self, x, y, z, validation_data=None, T_iter=250, batch_size=128,
          save_figs=False):
    n_sensitive = z.shape[1]
    if validation_data is not None:
      x_val, y_val, z_val = validation_data

    class_weight_adv = self._compute_class_weights(z)
    class_weight_clf_w_adv = [{0: 1., 1: 1.}] + class_weight_adv
    self._val_metrics = pd.DataFrame()
    self._fairness_metrics = pd.DataFrame()
    for idx in range(T_iter):
      print("Iteration index out of", [idx, T_iter])
      if validation_data is not None:
        # y_pred = pd.Series(self._clf.predict(x_val).ravel(), index=y_val.index)
        y_pred = self._clf.predict(x_val)
        self._val_metrics.loc[idx, 'ROC AUC'] = roc_auc_score(y_val, y_pred)
        self._val_metrics.loc[idx, 'Accuracy'] = (accuracy_score(y_val, (y_pred > 0.5)) * 100)
        # for sensitive_attr in z_val.columns:

        for sensitive_attr in range(n_sensitive):
          self._fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred,
                                                                   z_val[:,sensitive_attr])
        # display.clear_output(wait=True)
        # plot_distributions(y_pred, z_val, idx + 1, self._val_metrics.loc[idx],
        #                    self._fairness_metrics.loc[idx],
        #                    fname=f'output/{idx + 1:08d}.png' if save_figs else None)
        # plt.show(plt.gcf())

      # train adverserial
      self._trainable_clf_net(False)
      self._trainable_adv_net(True)
      # self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), batch_size=batch_size,
      #               class_weight=class_weight_adv, epochs=1, verbose=0)
      self._adv.fit(x, np.hsplit(z, z.shape[1]), batch_size=batch_size,
                    class_weight=class_weight_adv, epochs=1, verbose=0)

      # train classifier
      self._trainable_clf_net(True)
      self._trainable_adv_net(False)
      indices = np.random.permutation(len(x))[:batch_size]
      # self._clf_w_adv.train_on_batch(x.values[indices],
      #                                [y.values[indices]] + np.hsplit(z.values[indices], n_sensitive),
      #                                class_weight=class_weight_clf_w_adv)

      self._clf_w_adv.train_on_batch(x[indices],
                                     [y[indices]] + np.hsplit(z[indices], n_sensitive),
                                     class_weight=class_weight_clf_w_adv)


def load_data():

  # data, categorical_columns, ordinal_columns = load_dataset('adult')
  #
  # """ some specifics on this dataset """
  # numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
  # n_classes = 2
  #
  # data = data[:, numerical_columns + ordinal_columns + categorical_columns]
  #
  # num_numerical_inputs = len(numerical_columns)
  # num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1
  #
  # inputs = data[:, :-1]
  # target = data[:, -1]

  # X, y, Z = load_ICU_data('data/adult/adult.data')
  #
  # # split into train/test set
  # x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, y, Z, test_size=0.5,
  #                                                                      stratify=y, random_state=7)
  #
  # # standardize the data
  # scaler = StandardScaler().fit(x_train)
  # scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
  # x_train = x_train.pipe(scale_df, scaler)
  # x_test = x_test.pipe(scale_df, scaler)

  filename = 'code/adult.p'
  with open(filename, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    y_tot, x_tot = data

  # sort out Z from x_tot

  #
  # data = [
  #     age(0), workclass(1), fnlwgt(2), education(3), education_num(4),
  #     marital_status(5), occupation(6), relationship(7), race(8), sex(9),
  #     capital_gain(10), capital_loss(11), hours_per_week(12), native_country(13)]


  ind_race = 8
  ind_sex = 9

  race_feature = x_tot[:,ind_race]

  # I have take data for only white and black race.
  uval = np.unique(race_feature)
  ind_white = race_feature == uval[4]
  ind_black = race_feature == uval[2]

  x_tot = x_tot[ind_black + ind_white, :]
  y_tot = y_tot[ind_black+ind_white]

  race_feature = x_tot[:,ind_race]
  sex_feature = x_tot[:,ind_sex]

  # Z = (input_data.loc[:, sensitive_attribs]
  #      .assign(race=lambda df: (df['race'] == 'White').astype(int),
  #              sex=lambda df: (df['sex'] == 'Male').astype(int)))

  # binarize sensitive attributes
  Z = np.concatenate((np.expand_dims(race_feature,axis=1), np.expand_dims(sex_feature, axis=1)), axis=1) # two sensitive features

  ind_white = Z[:,0]==uval[4]
  Z[ind_white, 0] = 1 # White
  Z[~ind_white, 0] = 0 # rest

  uval = np.unique(Z[:, 1])
  ind_male = Z[:,1]==uval[1]
  Z[ind_male, 1] = 1 # Male
  Z[~ind_male, 1] = 0 # rest


  X = np.concatenate((x_tot[:,:8], x_tot[:,10:]),axis=1) # 12 input features
  y = y_tot

  # np.save('X_adult_for_fairness.npy', X)
  # np.save('y_adult_for_fairness.npy',y)


  x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, y, Z, test_size=0.5, stratify=y, random_state=7)

  return x_train, x_test, y_train, y_test, z_train, z_test


def train_baseline(x_train, x_test, y_train, y_test, z_train, z_test):
  # initialise NeuralNet Classifier
  base_clf = nn_classifier(n_features=x_train.shape[1])

  # train on train set
  # history = base_clf.fit(x_train.values, y_train.values, epochs=20, verbose=0)
  history = base_clf.fit(x_train, y_train, epochs=20, verbose=0)

  # y_pred = pd.Series(base_clf.predict(x_test).ravel(), index=y_test.index)
  y_pred = base_clf.predict(x_test)
  print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
  print(f"Accuracy: {100 * accuracy_score(y_test, (y_pred > 0.5)):.1f}%")

  # fig = plot_distributions(y_pred, z_test, fname='images/biased_training.png')
  #
  # print("The classifier satisfies the following %p-rules:")
  # print(f"\tgiven attribute race; {p_rule(y_pred, z_test['race']):.0f}%-rule")
  # print(f"\tgiven attribute sex;  {p_rule(y_pred, z_test['sex']):.0f}%-rule")

  extract_layers(base_clf, 'baseline_clf.npz')
  

def train_fair(x_train, x_test, y_train, y_test, z_train, z_test, T_iter):
  ##########################################
  # initialise FairClassifier
  fair_clf = FairClassifier(n_features=x_train.shape[1], n_sensitive=z_train.shape[1], lambdas=[130., 30.])

  print('starting to pretrain')
  # pre-train both adverserial and classifier networks
  fair_clf.pretrain(x_train, y_train, z_train, verbose=0, epochs=5)

  # if create_gif:
  #     !rm output/*.png
  print('starting to train')

  fair_clf.fit(x_train, y_train, z_train, validation_data=(x_test, y_test, z_test), T_iter=T_iter, save_figs=False)

  # To save fairness metrics and accuracy
  fairness_metrics = fair_clf._fairness_metrics
  fairness_metrics.to_pickle("fairness_metrics.pkl")

  eval_metrics = fair_clf._val_metrics
  eval_metrics.to_pickle("eval_metrics.pkl")


  # pred_on_0 = fair_clf._clf.predict(np.zeros(shape=(1, 94)))
  # pred_on_1 = fair_clf._clf.predict(np.ones(shape=(1, 94)))
  # print('pred after training', pred_on_0, pred_on_1)

  # change the number to include T_iter
  filename = 'fair_clf_' + str(T_iter) + '.npz'
  extract_layers(fair_clf._clf, filename)


def extract_layers(model, save_file):
  w_dict = dict()

  layer_idx = 0
  for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) != 0:
      assert len(weights) == 2 and len(weights[0].shape) == 2 and len(weights[1].shape) == 1
      w_dict[f'weight{layer_idx}'] = weights[0]
      w_dict[f'bias{layer_idx}'] = weights[1]
      layer_idx += 1

  print(f'saving the following weights: {w_dict.keys()}')
  np.savez(save_file, **w_dict)
  print("saving is done")


def main():
  data = load_data()
  
  # train_baseline(*data)

  train_fair(*data, T_iter=250)

  # train_fair(*data, T_iter=125)

  # train_fair(*data, T_iter=1)


if __name__ == '__main__':
  main()
