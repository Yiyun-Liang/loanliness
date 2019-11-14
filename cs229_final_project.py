import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE

from util import load_pickle_file
from util import report_test
from util import upsample_pos


def train_lr(x, y, rand_state=229, solver='liblinear',
        max_iter=10000, test=None):
    clf_lr = LogisticRegression(
        random_state=rand_state, solver=solver, max_iter=max_iter)
    clf_lr.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_lr, test, "logistic regression")
        return clf_lr, clf_acc
    return clf_lr



def train_rand_forest(x, y, n_est=100, max_depth=3, rand_state=229, test=None):
    clf_rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
        random_state=rand_state)
    clf_rf.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_rf, test, "random forest")
        return clf_rf, clf_acc
    return clf_rf

def train_nb(x, y, test=None):
    clf_nb = GaussianNB(var_smoothing=1e-7).fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_nb, test, "Gaussian Naive Bayes")
        return clf_nb, clf_acc
    return clf_nb


def train_mlp(x, y, solver='adam', alpha=1e-4, hls=(10, 40, 40),
        rand_state=229, test=None):
    clf_nn = MLPClassifier(
        solver=solver, alpha=alpha, hidden_layer_sizes=hls,
        random_state=rand_state)
    clf_nn.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_nn, test, "neural network")
        return clf_nn, clf_acc
    return clf_nn


if __name__ == '__main__':
    print('\n')
    training_data_path = 'training_data_processed.pkl'
    data = load_pickle_file(training_data_path)
    print('Training data has been successfully loaded')
    y = data[:, 1].astype(np.int)
    x = data[:, 2:]

    lr_acc_ls = []
    rf_acc_ls = []
    nb_acc_ls = []
    nn_acc_ls = []
    kf = KFold(n_splits=10, shuffle=True)
    print('Training is starting ... ')
    print(x.shape)
    x, y, x_test, y_test = upsample_pos(x, y, upsample=True)
    # for train, test in kf.split(x):
    # x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    # Logistic Regression
    # clf_lr, lr_acc = train_lr(x, y, test=[x_test, y_test])
    # lr_acc_ls.append(lr_acc)
    # Random Forest
    clf_rf, rf_acc = train_rand_forest(x, y, test=[x_test, y_test])
    rf_acc_ls.append(rf_acc)
    # Naive Bayes
    clf_nb, nb_acc = train_nb(x, y, test=[x_test, y_test])
    nb_acc_ls.append(nb_acc)
    # Neural Network
    clf_mlp, mlp_acc = train_mlp(x, y, test=[x_test, y_test])
    nn_acc_ls.append(mlp_acc)





