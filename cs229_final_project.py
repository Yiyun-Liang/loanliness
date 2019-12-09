import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans

from util import load_pickle_file
from util import save_pickle_file
from util import report_test
from util import upsample_pos
from util import data_preprocessing
from util import rand_train_test

def train_kmeans(x, y, test=None):
    kmeans = KMeans(n_clusters=2, random_state=229).fit(x)
    
    if test is not None:
        x_test, y_test = test
        # clf_acc = report_test(kmeans, test, "kmeans")
        # print(kmeans.cluster_centers_)
        y_pred = kmeans.predict(x_test)
        print((y_pred == y_test).sum()/len(y_test))
        return kmeans.labels_, y_pred
    return kmeans, test, 'kmeans'

def train_svm(x, y, kernel_type, test=None):
    clf_svm = SVC(kernel='linear', probability=True)
    if kernel_type == 'poly':
        clf_svm = SVC(kernel='poly', degree=8, probability=True)
    elif kernel_type == 'rbf':
        clf_svm = SVC(kernel='rbf', probability=True)
    elif kernel_type == 'sigmoid':
        clf_svm = SVC(kernel='sigmoid', probability=True)
    clf_svm.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_svm, test, "svm")
        return clf_svm, test, 'svm'
    return clf_svm, test, 'svm'

def train_lr(x, y, rand_state=229, solver='liblinear',
        max_iter=10000, test=None):
    clf_lr = LogisticRegression(
        random_state=rand_state, solver=solver, max_iter=max_iter, C=0.0001)
    # clf_lr = LogisticRegression(C = 0.0001)
    clf_lr.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_lr, test, "logistic regression")
        return clf_lr, clf_acc
    return clf_lr

def train_rand_forest(x, y, n_est=100, max_depth=3, rand_state=229, test=None):
    # clf_rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
    #     random_state=rand_state)
    clf_rf = RandomForestClassifier(n_estimators = 100, random_state = 50, n_jobs = -1)
    clf_rf.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_rf, test, "random forest")
        return clf_rf, clf_acc
    return clf_rf

def train_nb(x, y, test=None):
    clf_nb = GaussianNB().fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_nb, test, "Gaussian Naive Bayes")
        return clf_nb, clf_acc
    return clf_nb

def train_mlp(x, y, solver='lbfgs', alpha=1e-4, hls=(10, 40, 40),
        rand_state=229, test=None):
    clf_nn = MLPClassifier(
        solver=solver, alpha=alpha, hidden_layer_sizes=hls,
        random_state=rand_state)
    clf_nn.fit(x, y)
    if test is not None:
        clf_acc = report_test(clf_nn, test, "neural network")
        return clf_nn, clf_acc
    return clf_nn

def train_lgbm(x, y, test=None):
    clf_lgbm = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1, )
    clf_lgbm.fit(x, y, verbose=100)

    if test is not None:
        clf_acc = report_test(clf_lgbm, test, "LGBM")
        return clf_lgbm, clf_acc
    return clf_lgbm

if __name__ == '__main__':
    training_data_path = './data_preprocessed/training_data.pkl'
    label_path = './data_preprocessed/training_lbl.pkl'
    # training_data_path = './data_processed/training_data_processed.pkl'
    # label_path = './data_processed/training_lbl_processed.pkl'
    # training_data_path = './data_preprocessed/train_bureau_raw_data.pkl'
    # label_path = './data_preprocessed/train_bureau_raw_lbl.pkl'
    data = load_pickle_file(training_data_path)
    label = load_pickle_file(label_path)
    print('Training data has been successfully loaded')
    '''
    y = data[:, 1].astype(np.int)
    x = data[:, 2:]
    '''

    y = np.array(label)
    x = data
    # entries = list(data.columns)
    x = np.array(x)
    print(x.shape)
    # raise
    x, y = data_preprocessing(x, y, thres=0.1)

    lr_acc_ls = []
    rf_acc_ls = []
    nb_acc_ls = []
    nn_acc_ls = []
    lgbm_acc_ls = []
    # kf = KFold(n_splits=1, shuffle=True)
    print('Training is starting ... ')
    print('shape of x: {}'.format(x.shape))
    
    x, y, x_test, y_test = upsample_pos(x, y, upsample=False)
    # x, y, x_test, y_test = rand_train_test(x, y)
    # save_pickle_file(x, "training_data_up.pkl")
    # save_pickle_file(y, "training_lbl_up.pkl")
    # save_pickle_file(x_test, "testing_data_up.pkl")
    # save_pickle_file(y_test, "testing_lbl_up.pkl")
    # x = load_pickle_file('training_data_up.pkl')
    # y = load_pickle_file('training_lbl_up.pkl')
    # x_test = load_pickle_file('testing_data_up.pkl')
    # y_test = load_pickle_file('testing_lbl_up.pkl')
    # raise
    print('Percentage of zeros in trainset input: {}'.format(np.count_nonzero(x==0)/x.size))
    print('Number of positive examples: {}, negative: {}'.format((y==1).sum(), (y==0).sum()))
    # for train, test in kf.split(x):
    print("here")
    x_train, x_test, y_train, y_test = x, x_test, y, y_test
    print(x_train.shape)
    print(x_test.shape)
    print(len(y_test==1))
    print(len(y_test==0))
    # SVM
    # clf_svm, svm_acc = train_svm(x_train, y_train, kernel_type='linear', test=[x_test, y_test])
    # clf_svm, svm_acc = train_svm(x_train, y_train, kernel_type='poly', test=[x_test, y_test])
    # clf_svm, svm_acc = train_svm(x_train, y_train, kernel_type='rbf', test=[x_test, y_test])
    # clf_svm, svm_acc = train_svm(x_train, y_train, kernel_type='sigmoid', test=[x_test, y_test])

    # kmeans
    # NUM_CLUSTERS = 2
    # train_group, test_group = train_kmeans(x_train, y_train, test=[x_test, y_test])
    # for i in range(NUM_CLUSTERS):
    #     cur_train_x, cur_train_y = x_train[train_group==i], y_train[train_group==i]
    #     cur_test_x, cur_test_y = x_test[test_group==i], y_test[test_group==i]
    #     print('length of train set {}, test set {}'.format(len(cur_train_x), len(cur_test_x)))
    #     clf_lr, lr_acc = train_lr(cur_train_x, cur_train_y, test=[cur_test_x, cur_test_y])
    #     clf_rf, rf_acc = train_rand_forest(cur_train_x, cur_train_y, test=[cur_test_x, cur_test_y])
    # Logistic Regression
    clf_lr, lr_acc = train_lr(x_train, y_train, test=[x_test, y_test])
    lr_acc_ls.append(lr_acc)
    # Random Forest
    clf_rf, rf_acc = train_rand_forest(x_train, y_train, test=[x_test, y_test])
    rf_acc_ls.append(rf_acc)
    # # Naive Bayes
    clf_nb, nb_acc = train_nb(x_train, y_train, test=[x_test, y_test])
    nb_acc_ls.append(nb_acc)
    
    # # Neural Network
    clf_mlp, mlp_acc = train_mlp(x_train, y_train, test=[x_test, y_test])
    nn_acc_ls.append(mlp_acc)
    
    LGBMClassifier
    clf_lgbm, lgbm_acc = train_lgbm(x_train, y_train, test=[x_test, y_test])
    lgbm_acc_ls.append(lgbm_acc)





