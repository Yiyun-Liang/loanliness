import numpy as np
import csv
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import pandas as pd

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def count_empty_percentage(col):
    return Counter(col)['']/col.shape[0]

def count_nan_percentage(col):
    return Counter(col)['nan'] / col.shape[0]

def get_possible_values(col):
    return list(set(col))

def data_summary(data):
    # Number of features
    print('Number of features: '+str(data.shape[1]))
    print('Number of data points: '+str(data.shape[0]))
    return

def save_pickle_file(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def data_preprocessing(x, y, thres=0.1):
    # x = pd.DataFrame(data=x[:, 1:], index=x[:, 0])
    x = pd.DataFrame(data=x)
    x = x.replace([np.inf], 1e10)
    x = x.replace([-np.inf], -1e10)
    x = x.to_numpy()
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = imp.fit(x)
    # x = imp.transform(x)
    # print(x.shape)
    # print(y.shape)
    # save_pickle_file(x, "training_data_processed.pkl")
    # save_pickle_file(y, "training_lbl_processed.pkl")
    # raise

    # x = x[:, 1:]
    threshold = thres * len(x)
    should_keep = []
    for col in range(0, len(x[0])):
        curr_col = np.sum(np.isnan(x[:, col]))
        if curr_col <= threshold:
            should_keep.append(col)
    x = x[:, np.array(should_keep)]
    should_keep = []
    # np.where(x==np.inf, 1e10, x)
    for row in range(len(x)):
        # curr_mean = np.mean(x[:, col])
        # for row in range(len(x)):
        #     if np.isnan(x[row, col]):
        #         x[row, col] = curr_mean
        if np.sum(np.isnan(x[row, :])) == 0:
            should_keep.append(row)
    x = x[np.array(should_keep), :]
    y = y[np.array(should_keep)]
    # x[x >= 1e10] = 1e10
    # x[x <= -1 * 1e10] = -1 * 1e10
    print(x.shape)
    print(y.shape)
    save_pickle_file(x, "training_data_processed.pkl")
    save_pickle_file(y, "training_lbl_processed.pkl")
    # raise
    return x, y

def read_data(file_path):
    rows = []
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            rows.append(row)
    return rows

def get_processed_data(file_path, pkl_file_name):
    entries = np.array(read_data(file_path)[0])
    training_data = np.array(read_data(file_path)[1:])
    print('Training data has been loaded.')
    print('Training data preprocessing is starting.')
    training_data_processed, revised_entries, value_dict = data_preprocessing(entries, training_data)
    print('The summary of raw training data:')
    data_summary(training_data)
    print('The summary of processed training data:')
    data_summary(training_data_processed)
    save_pickle_file(training_data_processed, pkl_file_name)
    return

def report_test(clf, test, clf_name):
    x_test, y_test = test
    clf_acc = clf.score(x_test, y_test)
    print('The accuracy for ' + clf_name + ' classifier is: '+ str(clf_acc))
    y_pred = clf.predict(x_test)
    print("Prediction Positive Number: " + str(np.sum(y_pred == 1)) + " True Number: " + str(np.sum(y_test == 1)))
    print("Prediction Negative Number: " + str(np.sum(y_pred == 0)) + " True Number: " + str(np.sum(y_test == 0)))
    print(classification_report(y_test, y_pred))
    return clf_acc

def upsample_pos(x, y, upsample=True):
    # less positive, more negative
    all_pos = np.where(y == 1)
    print(len(all_pos[0]))
    x_all_pos = x[all_pos[0]]
    y_all_pos = y[all_pos[0]]
    cut_len = len(x_all_pos) // 5
    x_test = x_all_pos[:cut_len]
    y_test = y_all_pos[:cut_len]
    x_all_pos = x_all_pos[cut_len + 1:]
    y_all_pos = y_all_pos[cut_len + 1:]

    all_neg = np.where(y == 0)
    print(len(all_neg[0]))
    x_all_neg = x[all_neg[0]]
    y_all_neg = y[all_neg[0]]
    x_test = np.concatenate((x_test, x_all_neg[:cut_len]), axis=0)
    y_test = np.concatenate((y_test, y_all_neg[:cut_len]), axis=0)
    x_all_neg = x_all_neg[cut_len + 1:]
    y_all_neg = y_all_neg[cut_len + 1:]

    if upsample:
        rand_ind = np.arange(len(x_all_neg))
        np.random.shuffle(rand_ind)
        x_neg_new = x_all_neg[rand_ind[:2*len(x_all_pos)]]
        y_neg_new = y_all_neg[rand_ind[:2*len(x_all_pos)]]
        x_all_new = np.concatenate((x_neg_new, x_all_pos), axis=0)
        y_all_new = np.concatenate((y_neg_new, y_all_pos), axis=0)
        sm = SMOTE(random_state=233333, sampling_strategy=1.0, k_neighbors=1000)
        x_train, y_train = sm.fit_sample(x_all_new, y_all_new)
    else:
        # undersample: balance train set
        x_all_neg = x_all_neg[:int(len(x_all_pos))]
        y_all_neg = y_all_neg[:int(len(x_all_pos))]
        x_train = np.concatenate((x_all_neg, x_all_pos), axis=0)
        y_train = np.concatenate((y_all_neg, y_all_pos), axis=0)
    
    rand_shuffle = np.arange(len(x_train))
    np.random.shuffle(rand_shuffle)
    x_train = x_train[rand_shuffle]
    y_train = y_train[rand_shuffle]
    
    rand_shuffle_test = np.arange(len(x_test))
    np.random.shuffle(rand_shuffle_test)
    x_test = x_test[rand_shuffle_test]
    y_test = y_test[rand_shuffle_test]
    return x_train, y_train, x_test, y_test


def rand_train_test(x, y):
    cut_len = len(x) // 5
    rand_ind = np.arange(len(x))
    np.random.shuffle(rand_ind)
    x_train = x[rand_ind[cut_len:]]
    y_train = y[rand_ind[cut_len:]]
    x_test = x[rand_ind[:cut_len]]
    y_test = y[rand_ind[:cut_len]]
    return x_train, y_train, x_test, y_test







