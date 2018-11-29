import pandas as pd
import numpy.random as nr
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn import ensemble
from sklearn import linear_model
from sklearn import neural_network
from sklearn.svm import LinearSVC

from sklearn.preprocessing import LabelEncoder


# funkcija koja traži najbolje hiperparametre gradient boostinga
def gbHyperparametars(X_train, y_train):
    algorithmParameters = dict()  # dict hiperparametara koje ćemo testirati

    # popunjavanje algorithmParameters
    nEstimatorsValues = []
    for i in range(3, 6):
        nEstimatorsValues.append(4 ** i)

    nEstimatorsValues.append(100)

    algorithmParameters['n_estimators'] = nEstimatorsValues

    learningRateValues = []
    for i in range(3):
        learningRateValues.append(0.01 * 10 ** i)

    algorithmParameters['learning_rate'] = learningRateValues

    algorithmParameters['max_depth'] = [2, 4, 6]

    model = ensemble.GradientBoostingClassifier()

    # traženje i ispis najboljih hiperparametara
    grid = GridSearchCV(model, algorithmParameters,
                        scoring=metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True))
    print('searching...')
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.n_estimators)
    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.learning_rate)

    classifierGB = ensemble.GradientBoostingClassifier(n_estimators=grid.best_estimator_.n_estimators,
                                                       max_depth=grid.best_estimator_.max_depth,
                                                       learning_rate=grid.best_estimator_.learning_rate)

    return classifierGB


# funkcija koja traži najbolje hiperparametre random forest
def rfHyperparametars(X_train, y_train, n_estimators):
    algorithmParameters = dict()

    algorithmParameters['min_samples_split'] = [2, 4, 8]

    algorithmParameters['max_features'] = ['auto', 0.0001, 0.005, 0.01, 0.1]

    algorithmParameters['max_depth'] = [None, 4, 10, 32]

    algorithmParameters['criterion'] = ['gini', "entropy"]

    algorithmParameters['class_weight'] = ['balanced', None]

    model = ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

    grid = GridSearchCV(model, algorithmParameters, scoring=metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True))
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.max_depth)
    print(grid.best_estimator_.min_samples_split)
    print(grid.best_estimator_.max_features)
    print(grid.best_estimator_.criterion)
    print(grid.best_estimator_.class_weight)

    classifierRF = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                   min_samples_split=grid.best_estimator_.min_samples_split,
                                                   max_depth=grid.best_estimator_.max_depth,
                                                   max_features=grid.best_estimator_.max_features,
                                                   criterion=grid.best_estimator_.criterion,
                                                   class_weight=grid.best_estimator_.class_weight)

    return classifierRF


# hiperparametri za logističku regresiju
def lrHyperparametars(X_train, y_train):
    algorithmParameters = dict()

    algorithmParameters['penalty'] = ['l1', 'l2']

    CValues = []
    for i in range(1, 10):
        CValues.append(0.001 * 2 ** i)

    algorithmParameters['C'] = CValues

    # algorithmParameters['dual'] = [True, False]

    model = linear_model.LogisticRegression(class_weight='balanced')

    grid = GridSearchCV(model, algorithmParameters, scoring=metrics.make_scorer(metrics.roc_auc_score))
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.penalty)
    # print(grid.best_estimator_.dual)
    print(grid.best_estimator_.C)

    classifierLR = linear_model.LogisticRegression(penalty=grid.best_estimator_.penalty, C=grid.best_estimator_.C,
                                                   class_weight='balanced')
    # dual = grid.best_estimator_.dual)

    return classifierLR


def svcHyperparametars(X_train, y_train):
    algorithmParameters = dict()

    CValues = []
    for i in range(1, 15):
        CValues.append(0.001 * 2 ** i)

    algorithmParameters['C'] = CValues

    model = LinearSVC(class_weight='balanced')

    grid = GridSearchCV(model, algorithmParameters)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_.penalty)
    print(grid.best_estimator_.C)

    classifierSVC = LinearSVC(C=grid.best_estimator_.C)  # dual = False

    return classifierSVC

