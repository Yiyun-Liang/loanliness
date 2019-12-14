# from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from util import save_pickle_file
from util import load_pickle_file

def load_data(path):
	data = pd.read_csv(path)
	train_labels = data['TARGET']
	data = data.drop(columns = ['TARGET'])
	return data, train_labels


def _lbl_encode(data):
	le = LabelEncoder()
	for col in data:
	    if data[col].dtype == 'object':
	        if len(list(data[col].unique())) <= 2:
	            le.fit(data[col])
	            data[col] = le.transform(data[col])
	return data

def _onehot_encode(data):
	data = pd.get_dummies(data)
	return data

def encode_data(data):
	data = _lbl_encode(data)
	data = _onehot_encode(data)
	return data

def remove_days_abnormal(data):
	data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243
	data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
	return data

def merge_poly_features(data):
	poly_features = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
	from sklearn.preprocessing import Imputer
	imputer = Imputer(strategy = 'median')
	poly_features = imputer.fit_transform(poly_features)
	poly_features_test = imputer.transform(poly_features_test)
	from sklearn.preprocessing import PolynomialFeatures
	poly_transformer = PolynomialFeatures(degree = 3)
	poly_transformer.fit(poly_features)
	poly_features = poly_transformer.transform(poly_features)
	poly_features = pd.DataFrame(poly_features, 
	                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
	                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
	poly_features['SK_ID_CURR'] = data['SK_ID_CURR']
	data_poly = data.merge(poly_features, on = 'SK_ID_CURR', how = 'left')
	return data_poly

def agg_numeric(df, group_var, df_name):
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    columns = [group_var]
    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    return agg

def count_categorical(df, group_var, df_name):
    categorical = pd.get_dummies(df.select_dtypes('object'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    column_names = []
    for var in categorical.columns.levels[0]:
        for stat in ['count', 'count_norm']:
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    categorical.columns = column_names
    return categorical

def _bureau_helper(data):
	data_counts = count_categorical(data, group_var = 'SK_ID_CURR', df_name = str(data))
	data_agg = agg_numeric(data.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
	return data_agg

def merge_bureau(data):
	bureau = pd.read_csv('../input/bureau.csv')
	bureau_balance = pd.read_csv('../input/bureau_balance.csv')
	bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
	bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
	bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
	bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
	bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
	bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
	bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
	data = data.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
	data = data.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
	data = data.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
	return data

def agg_numeric_new(df, parent_var, df_name):
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    columns = []
    for var in agg.columns.levels[0]:
        if var != parent_var:
            for stat in agg.columns.levels[1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    return agg


def agg_categorical_new(df, parent_var, df_name):
    categorical = pd.get_dummies(df.select_dtypes('category'))
    categorical[parent_var] = df[parent_var]
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])
    column_names = []
    for var in categorical.columns.levels[0]:
        for stat in ['sum', 'count', 'mean']:
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    categorical.columns = column_names
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    return categorical


def aggregate_client(df, group_vars, df_names):
    df_agg = agg_numeric_new(df, parent_var = group_vars[0], df_name = df_names[0])
    if any(df.dtypes == 'category'):
        df_counts = agg_categorical_new(df, parent_var = group_vars[0], df_name = df_names[0])
        df_by_loan = df_counts.merge(df_agg, on = group_vars[0], how = 'outer')
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric_new(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])

        
    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
        
        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])
        
        # Aggregate numeric stats by column
        df_by_client = agg_numeric_new(df_by_loan, parent_var = group_vars[1], df_name = df_names[1])
        
    # Memory management

    return df_by_client

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df




