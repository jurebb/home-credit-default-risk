# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
                            make_scorer
from sklearn.model_selection import train_test_split
import numpy as np

DIR_JURE = '/media/interferon/44B681D7B681C9BE/kaggle/home-credit-default\
-risk-data'
def init():
    os.chdir(DIR_JURE)
    df_appl_test = pd.read_csv('application_test.csv')
    df_appl_train = pd.read_csv('application_train.csv')
    return df_appl_train, df_appl_test


# Provjeri koj isu stupci categorical i koje vrijednosti primaju
def data_analysis(df_app_train):
    df_x = df_app_train.loc[:, df_app_train.columns != 'TARGET']

    for i in range(len(df_x.dtypes)):
        if df_x.dtypes[i] == object:
            print('===============')
            print('categorical col', df_x.columns[i])
            print('unique vals', set(list(df_x[df_x.columns[i]])))


def convert_cat_to_numer(df):
    for col_name in df.columns:
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes


def simple_baseline(df_app_train, df_app_test):
    print('> simple_baseline')
    print('df_app_train.shape', df_app_train.shape)
    print('df_app_test.shape', df_app_test.shape)
    
    convert_cat_to_numer(df_app_train)
    convert_cat_to_numer(df_app_test)
    
    df_x = df_app_train.loc[:, df_app_train.columns != 'TARGET']
    df_y = df_app_train['TARGET']
    
    df_x = df_x.as_matrix().astype(np.float)
    df_y = df_y.as_matrix().astype(np.float)
    
    df_x = np.nan_to_num(df_x)
    df_y = np.nan_to_num(df_y)
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                test_size=0.2, random_state=42)
    
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print('acc', accuracy_score(pred, y_test))
    print('f1', f1_score(pred, y_test, average = None))
    print('f1_macro', f1_score(pred, y_test, average = 'macro'))


# IDEA: for previous Credit Bureau credits of a loan from df_app_train, we have the balances in bureau_balance.csv
# LSTM: balance values as inputs (maybe masking different length inputs) and the TARGET values from df_app_train
# as desired outputs?
def bureau_recurrent(df_app_train):
    print('> bureau_recurrent')
    print('df_app_train.shape', df_app_train.shape)
    df_bureau = pd.read_csv('bureau.csv')
    df_bureau_balance = pd.read_csv('bureau_balance.csv')
    print('df_bureau.shape', df_bureau.shape)
    print('df_bureau_balance.shape', df_bureau_balance.shape)

    # These are ID and target values for current loans from the main application_train.csv
    df_id_y = df_app_train.loc[:, 'SK_ID_CURR':'TARGET']

    # df [bureau cols - target]
    df_joined_bureau_id = df_bureau.merge(df_id_y, on='SK_ID_CURR', how='left')

    # we don't need this anymore
    del df_id_y

    # target values
    df_y = df_joined_bureau_id['TARGET']

    df_x = df_joined_bureau_id.loc[:, df_joined_bureau_id.columns != 'TARGET']

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                        test_size=0.2, random_state=42)




if __name__ == "__main__":
    df_app_train, df_app_test = init()
    #data_analysis(df_app_train)
    #simple_baseline(df_app_train, df_app_test)

    bureau_recurrent(df_app_train)

    exit(0)
    