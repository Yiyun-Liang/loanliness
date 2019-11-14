import numpy as np
import csv
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
import sklearn

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def count_empty_percentage(col):
    return Counter(col)['']/col.shape[0]

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

def data_preprocessing(entries, data, test = False):
    value_dict = dict()               # Store the possible values of the col
    processed_data = np.copy(data)
    if not test:
        start_entry = 2
    else:
        start_entry = 1
    remove_row_ls = []
    remove_col_ls = []
    for i in range(start_entry, data.shape[1]):
        # Assign values to string elements
        col = data[:, i]
        possible_values = sorted(get_possible_values(col))
        if '' in possible_values:
            possible_values.remove('')
        if len(possible_values) < 100:
            value_dict[entries[i]] = possible_values
            for j in range(data.shape[0]):
                if data[j, i] != '':
                    value = possible_values.index(data[j, i])
                    processed_data[j, i] = value
    print('Preprocessing: value assignment has been finished.')
    for i in range(start_entry, data.shape[1]):
        # Add the column with too many empty entries to remove_col_ls
        col = data[:, i]
        if count_empty_percentage(col) > 0.1:
            remove_col_ls.append(i)
        # Add the row with empty entries to remove_row_ls
        elif count_empty_percentage(col) != 0:
            for j in range(data.shape[0]):
                if data[j, i] == '':
                    remove_row_ls.append(j)
    print('Preprocessing: removal row and col numbers has been stored.')
    # Remove the data points and features which do not satisfy requiremetns
    remove_col_set = set(remove_col_ls)
    remove_row_set = set(remove_row_ls)
    print(len(remove_row_set))
    revised_entries = np.copy(entries)
    processed_data = np.delete(processed_data, list(remove_col_set), 1)
    print('Preprocessing: cols have been removed.')
    processed_data = np.delete(processed_data, list(remove_row_set), 0)
    print('Preprocessing: rows have been removed.')
    np.delete(revised_entries, list(remove_col_set))
    print('Preprocessing: preprocessing has been finished.')
    return processed_data.astype(np.float), revised_entries, value_dict


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
    return clf_acc

def upsample_pos(x, y, upsample=True):
    x = sklearn.preprocessing.normalize(x)
    # print(x[:2])
    # raise
    all_pos = np.where(y == 1)
    x_all_pos = x[all_pos[0]]
    y_all_pos = y[all_pos[0]]
    cut_len = len(x_all_pos) // 20
    x_test = x_all_pos[:cut_len]
    y_test = y_all_pos[:cut_len]
    x_all_pos = x_all_pos[cut_len + 1:]
    y_all_pos = y_all_pos[cut_len + 1:]
    all_neg = np.where(y == 0)
    x_all_neg = x[all_neg[0]]
    y_all_neg = y[all_neg[0]]
    x_test = np.concatenate((x_test, x_all_neg[:cut_len]), axis=0)
    y_test = np.concatenate((y_test, y_all_neg[:cut_len]), axis=0)
    x_all_neg = x_all_neg[cut_len + 1:]
    y_all_neg = y_all_neg[cut_len + 1:]
    if upsample:
        rand_ind = np.arange(len(x_all_neg))
        np.random.shuffle(rand_ind)
        x_neg_new = x_all_neg[rand_ind[:10*len(x_all_pos)]]
        y_neg_new = y_all_neg[rand_ind[:10*len(x_all_pos)]]
        x_all_new = np.concatenate((x_neg_new, x_all_pos), axis=0)
        y_all_new = np.concatenate((y_neg_new, y_all_pos), axis=0)
        sm = SMOTE(random_state=233333, sampling_strategy='minority', k_neighbors=100)
        x_train, y_train = sm.fit_sample(x_all_new, y_all_new)
    else:
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





