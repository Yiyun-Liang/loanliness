import csv
import numpy as np
from collections import Counter
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE

def read_data(file_path):
    rows = []
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            rows.append(row)
    return rows

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

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

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

if __name__ == '__main__':

    print('\n')
    training_data_path = 'training_data_processed.pkl'
    data = load_pickle_file(training_data_path)
    print('Training data has been successfully loaded')
    y = data[:, 1].astype(np.int)
    X = data[:, 2:]

    lr_acc_ls = []
    random_forest_acc_ls = []
    nb_acc_ls = []
    nn_acc_ls = []
    svm_acc_ls = []
    kf = KFold(n_splits=10, shuffle=True)
    iter = 1
    print('Training is starting ... ')
    print(X_new.shape)
    for train, test in kf.split(X_new):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        # X_train, X_test, y_train, y_test = X_new[train], X_new[test], y_new[train], y_new[test]

        print('Iteration '+ str(iter))

        # Logistic Regression
        clf_lr = LogisticRegression(random_state=0, solver='lbfgs', max_iter=5000, multi_class='multinomial').fit(X_train, y_train)
        clf_lr.fit(X_train, y_train)
        lr_acc = clf_lr.score(X_test, y_test)
        lr_acc_ls.append(lr_acc)
        print('The accuracy for logistic regression classifier is: '+str(lr_acc))

        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)
        rf_acc = clf_rf.score(X_test, y_test)
        random_forest_acc_ls.append(rf_acc)
        print('The accuracy for random forest classifier is: ' + str(rf_acc))

        # Naive Bayes
        clf_nb = GaussianNB().fit(X_train, y_train)
        nb_acc = clf_nb.score(X_test, y_test)
        nb_acc_ls.append(nb_acc)
        print('The accuracy for Gaussian Naive Bayes classifier is: ' + str(nb_acc))

        # Neural Network
        clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (10, 10), random_state = 1).fit(X_train, y_train)
        nn_acc = clf_nn.score(X_test, y_test)
        nn_acc_ls.append(nn_acc)
        print('The accuracy for neural network classifier is: ' + str(nn_acc))

        '''
        # SVM
        clf_svm = SVC(gamma='auto', max_iter = 500).fit(X_train, y_train)
        svm_acc = clf_svm.score(X_test, y_test)
        svm_acc_ls.append(svm_acc)
        print('The accuracy for SVM classifier is: ' + str(svm_acc))
        '''
        print('\n')
        iter += 1

    print('The average accuracy for logistic regression classifier is: ' + str(sum(lr_acc_ls)/len(lr_acc_ls)))
    #print('The average accuracy for SVM classifier is: ' + str(sum(svm_acc_ls) / len(svm_acc_ls)))
    print('The average accuracy for Gaussian Naive Bayes classifier is: ' + str(sum(nb_acc_ls) / len(nb_acc_ls)))
    print('The average accuracy for neural network classifier is: ' + str(sum(nn_acc_ls) / len(nn_acc_ls)))
    print('The average accuracy for random forest classifier is: ' + str(sum(random_forest_acc_ls) / len(random_forest_acc_ls)))