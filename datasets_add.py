from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyper import *
import os


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


def add_set(filename, df_appl_train, suffix):

    df = pd.read_csv(filename)

    za_dodat = []
    x_id = df['SK_ID_CURR'].as_matrix()
    x = df.as_matrix()
    videni_id = []
    for i in range(df.shape[0]):
        if x_id[i] in df_appl_train['SK_ID_CURR'].as_matrix() and x_id[i] not in videni_id:
            videni_id.append(x_id[i])
            za_dodat.append(i)

    df_konacni = df_appl_train.join(df.iloc[za_dodat].set_index('SK_ID_CURR'), on='SK_ID_CURR', rsuffix=suffix)

    print(df_konacni.head())

    return df_konacni


dir = 'all'
os.chdir(dir)
df_appl_train = pd.read_csv('application_train.csv')

df_konacni = add_set('previous_application.csv', df_appl_train, 'prev_app_')
df_konacni = add_set('POS_CASH_balance.csv', df_konacni, 'POS_CASH_')
df_konacni = add_set('installments_payments.csv', df_konacni, 'install_payments_')
df_konacni = add_set('credit_card_balance.csv', df_konacni, 'balance_')

df_konacni = convert_cat_to_numer2(df_konacni)

df_x = df_konacni.loc[:, df_konacni.columns != 'TARGET']
df_y = df_konacni['TARGET']

nans = np.isnan(df_x)

df_x = df_x.fillna(-0.1013)
means = np.mean(df_x.as_matrix(), axis=0)

df_x = df_x.as_matrix().astype(np.float)
df_y = df_y.as_matrix().astype(np.float)

for i in range(df_x.shape[0]):
    for j in range(df_x.shape[1]):
        if df_x[i][j] == -0.1013:
            df_x[i][j] = means[j]
            #print('blin')

#df_x = np.nan_to_num(df_x)  # pogledat moze li bolje
#df_y = np.nan_to_num(df_y)  # pogledat moze li bolje

df_x = np.concatenate((df_x, nans), axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                    test_size=0.2, random_state=42)

mm_scaler = StandardScaler()
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


