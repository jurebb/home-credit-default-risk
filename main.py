# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
    make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from hyper import *

DIR_JURE = 'all'


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


# one hot encoding
def convert_cat_to_numer(df):
    col_names = []
    for col_name in df.columns:
        if (df[col_name].dtype == 'object'):
            col_names.append(col_name)

    df = pd.get_dummies(df, columns=col_names)

    return df


# label encoding
def convert_cat_to_numer2(df):
    for col_name in df.columns:
        if (df[col_name].dtype == 'object'):
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    return df


def simple_baseline(df_app_train, df_app_test):
    print('> simple_baseline')
    print('df_app_train.shape', df_app_train.shape)
    print('df_app_test.shape', df_app_test.shape)

    df_app_train = convert_cat_to_numer(df_app_train)
    df_app_test = convert_cat_to_numer(df_app_test)

    print(df_app_train.shape)

    df_x = df_app_train.loc[:, df_app_train.columns != 'TARGET']
    df_y = df_app_train['TARGET']

    df_x = df_x.as_matrix().astype(np.float)
    df_y = df_y.as_matrix().astype(np.float)

    df_x = np.nan_to_num(df_x)  # pogledat moze li bolje
    df_y = np.nan_to_num(df_y)  # pogledat moze li bolje

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                        test_size=0.2, random_state=42)

    mm_scaler = MinMaxScaler()
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)

    lgr = linear_model.LogisticRegression(C=0.05, penalty='l1')
    lgr.fit(x_train, y_train)
    pred = lgr.predict(x_test)
    print(pred)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('roc: ', metrics.roc_auc_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test, average='macro'))

    rf = rfHyperparametars(x_train, y_train, 20)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    print(pred)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('roc: ', metrics.roc_auc_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test, average='macro'))


    gb = ensemble.GradientBoostingClassifier()
    gb.fit(x_train, y_train)
    pred = gb.predict(x_test)
    print(pred)
    print('acc: ', metrics.accuracy_score(pred, y_test))
    print('roc: ', metrics.roc_auc_score(pred, y_test))
    print('f1: ', metrics.f1_score(pred, y_test, average='macro'))


if __name__ == "__main__":
    df_app_train, df_app_test = init()
    # data_analysis(df_app_train)
    simple_baseline(df_app_train, df_app_test)
    exit(0)
