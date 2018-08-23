# -*- coding: utf-8 -*-
import baseline
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
                            make_scorer
from sklearn.model_selection import train_test_split
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from keras import backend as K

import pickle

DIR_JURE = '/media/interferon/44B681D7B681C9BE/kaggle/home-credit-default\
-risk-data'

# reset weights on keras model
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}'.format(layer.name, v))


# data preparation function (no need to call if .pickle data exists)
def bureau_recurrent_data_init(df_app_train):
    print('> bureau_recurrent_data_init')
    print('df_app_train.shape', df_app_train.shape)
    df_bureau = pd.read_csv('bureau.csv')
    df_bureau_balance = pd.read_csv('bureau_balance.csv')
    print('df_bureau.shape', df_bureau.shape)
    print('df_bureau_balance.shape', df_bureau_balance.shape)

    # These are ID and target values for current loans from the main application_train.csv
    df_id_y = df_app_train.loc[:, 'SK_ID_CURR':'TARGET']

    # df [bureau cols - target]
    df_bureau_id = df_bureau.merge(df_id_y, on='SK_ID_CURR', how='left')

    # we don't need this anymore
    del df_id_y
    del df_app_train

    # df cols:[bureau cols - target - bureau_balance cols]
    df_joined_bureau_id = df_bureau_id.merge(df_bureau_balance, on='SK_ID_BUREAU', how='left')

    del df_bureau
    del df_bureau_id
    del df_bureau_balance

    print('df_joined_bureau_id.columns', df_joined_bureau_id.columns)
    print('df_joined_bureau_id.shape', df_joined_bureau_id.shape)

    # remove nan values from STATUS
    df_joined_bureau_id = df_joined_bureau_id[pd.notnull(df_joined_bureau_id['STATUS'])]

    print('df_joined_bureau_id.shape, after NaN row removal from STATUS', df_joined_bureau_id.shape)

    # target values (id to be removed later)
    df_y = df_joined_bureau_id.loc[:, ['SK_ID_BUREAU', 'TARGET']]

    # lstm input values, need reformatting...
    df_x = df_joined_bureau_id.loc[:, ['SK_ID_BUREAU', 'STATUS']]

    print('df_y.shape', df_y.shape)
    print('df_x.shape', df_x.shape)

    del df_joined_bureau_id

    # each unique SK_ID_BUREAU presents a single sequential example of STATUS values. Desired output is TARGET for a
    # given unique SK_ID_BUREAU
    dataset = []
    targets = []

    unique_ids = df_x['SK_ID_BUREAU'].unique()
    counter = 0
    length = len(unique_ids)
    for id in unique_ids:
        status_vals_for_current_id = df_x.loc[df_x['SK_ID_BUREAU'] == id]
        temp = np.array([])
        for _, status in status_vals_for_current_id.iteritems():
            temp = np.append(temp, status)

        target_val_for_current_id = df_y.loc[df_x['SK_ID_BUREAU'] == id].iloc[0].loc['TARGET']

        dataset.append(temp)
        targets.append(target_val_for_current_id)

        counter += 1

        if counter % 5000 == 0:
            print(counter, 'of', length)

    print('len(dataset)', len(dataset))
    print('len(targets)', len(targets))

    print('dataset', dataset)
    print('targets', targets)

    # Saving the objects for later use
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('bureau_targets.pickle', 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('df_x.pickle', 'wb') as handle:
        pickle.dump(df_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('df_y.pickle', 'wb') as handle:
        pickle.dump(df_y, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Further data preparation
# data preparation function (no need to call if .pickle data exists)
def bureau_recurrent_data_preparation():
    print('> bureau_recurrent_data_preparation')
    # Load the created data from bureau_recurrent_data_init

    with open('dataset.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    with open('bureau_targets.pickle', 'rb') as handle:
        targets = pickle.load(handle)

    # dataset[1] = list(filter((dataset[1][0]).__ne__, dataset[1]))
    indices_list = []

    length = len(dataset)
    # Remove the repeating SK_ID_BUREAU from the list and append it to a new list which we will save
    for i in range(length):

        indices_list.append(dataset[i][0])

        if len(str(dataset[i][0])) < 2:
            print('error, not id in question:', dataset[i][0], dataset[i], i)
            exit(1)

        dataset[i] = list(filter((dataset[i][0]).__ne__, dataset[i]))

        if i % 5000 == 0:
            print(i, 'of', length)

    with open('bureau_dataset_no_indices.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('bureau_indices_list.pickle', 'wb') as handle:
        pickle.dump(indices_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


# IDEA: for previous Credit Bureau credits of a loan from df_app_train, we have the balances in bureau_balance.csv
# LSTM: balance values as inputs (maybe masking different length inputs) and the TARGET values from df_app_train
# as desired outputs?
def bureau_recurrent_model():
    print('> bureau_recurrent_model')
    # Load the created data from bureau_recurrent_data

    with open('bureau/bureau_targets.pickle', 'rb') as handle:
        bureau_targets = pickle.load(handle)

    with open('bureau/bureau_dataset_no_indices.pickle', 'rb') as handle:
        bureau_dataset = pickle.load(handle)

    with open('bureau/bureau_indices_list.pickle', 'rb') as handle:
        bureau_indices_list = pickle.load(handle)

    # Remove nans from bureau_targets (and coresponding indices from other lists)
    df = pd.DataFrame(columns=['bureau_targets'], data=bureau_targets)
    df['bureau_dataset'] = pd.Series(bureau_dataset)
    df['bureau_indices_list'] = pd.Series(bureau_indices_list)
    print('dataset.shape before NaN removal:', df.shape)

    df = df[np.isfinite(df['bureau_targets'])]
    print('dataset.shape:', df.shape)
    print('dataset.columns:', df.columns)

    # Indices will be necessary for later integration into other datasets
    bureau_targets = df['bureau_targets'].values
    bureau_dataset = df['bureau_dataset'].values
    bureau_indices_list = df['bureau_indices_list'].values
    del df

    # find all possible categorical values in bureau_dataset and find the longest one
    all_possible_values = set()
    maxlen = 0
    for ex in bureau_dataset:
        all_possible_values.update(ex)
        if len(ex) > maxlen:
            maxlen = len(ex)

    # integer encode the categorical values of 'bureau_dataset'
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list(all_possible_values))
    print('all_possible_values, integer_encoded', all_possible_values, integer_encoded)
    print('maximum sample length', maxlen)

    for i in range(len(bureau_dataset)):
        bureau_dataset[i] = label_encoder.transform(bureau_dataset[i])

    # binary encode
    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # # invert first example
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)

    # Changing the datatype to integer (from float)
    bureau_targets = bureau_targets.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(bureau_dataset, bureau_targets,
                                                        test_size=0.2, random_state=42)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # TODO masking
    # TODO Embedding? without?
    # TODO Dropout?

    model = Sequential()
    model.add(Embedding(10, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=1024, epochs=1)
    score = model.evaluate(x_test, y_test, batch_size=1024)

    print('model.evaluate', score)

    y_pred = model.predict_classes(x_test)

    print('classification report:\n', classification_report(y_test, y_pred))


# VARIANT 2 of the above function
# Without using embedding in first layer
def bureau_recurrent_model_no_embedding():
    print('> bureau_recurrent_model')
    # Load the created data from bureau_recurrent_data

    with open('bureau/bureau_targets.pickle', 'rb') as handle:
        bureau_targets = pickle.load(handle)

    with open('bureau/bureau_dataset_no_indices.pickle', 'rb') as handle:
        bureau_dataset = pickle.load(handle)

    with open('bureau/bureau_indices_list.pickle', 'rb') as handle:
        bureau_indices_list = pickle.load(handle)

    # Remove nans from bureau_targets (and coresponding indices from other lists)
    df = pd.DataFrame(columns=['bureau_targets'], data=bureau_targets)
    df['bureau_dataset'] = pd.Series(bureau_dataset)
    df['bureau_indices_list'] = pd.Series(bureau_indices_list)
    print('dataset.shape before NaN removal:', df.shape)

    df = df[np.isfinite(df['bureau_targets'])]
    print('dataset.shape:', df.shape)
    print('dataset.columns:', df.columns)

    # Indices will be necessary for later integration into other datasets
    bureau_targets = df['bureau_targets'].values
    bureau_dataset = df['bureau_dataset'].values
    bureau_indices_list = df['bureau_indices_list'].values
    del df

    # find all possible categorical values in bureau_dataset and find the longest one
    all_possible_values = set()
    maxlen = 0
    for ex in bureau_dataset:
        all_possible_values.update(ex)
        if len(ex) > maxlen:
            maxlen = len(ex)

    # integer encode the categorical values of 'bureau_dataset'
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list(all_possible_values))
    print('all_possible_values, integer_encoded', all_possible_values, integer_encoded)
    print('maximum sample length', maxlen)

    for i in range(len(bureau_dataset)):
        bureau_dataset[i] = label_encoder.transform(bureau_dataset[i])

    # binary encode
    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # # invert first example
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)

    # Changing the datatype to integer (from float)
    bureau_targets = bureau_targets.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(bureau_dataset, bureau_targets,
                                                        test_size=0.2, random_state=42)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # TODO masking
    # TODO Embedding? without?
    # TODO Dropout?
    # TODO Shuffle between epochs?

    x_train = x_train.reshape(418812, 97, 1)
    x_test = x_test.reshape(104703, 97, 1)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    learning_rate = 0.0001
    dropout_ratio = 0.2
    class_0_weight = 1
    class_1_weight = 11.5

    # weights for the training samples, used for weighting the loss function (during training only)
    sample_weights = np.array([class_0_weight if y == 0 else class_1_weight for y in y_train])

    model = Sequential()
    # model.add(Embedding(10, 32))
    # batch_size, timesteps, data_dim
    model.add(LSTM(64, input_shape=(97, 1)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dropout(dropout_ratio))

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=30,
              validation_split=0.1,
              verbose=1,
              sample_weight=sample_weights,
              callbacks=[earlystopper],
              shuffle=True)

    score = model.evaluate(x_test, y_test, batch_size=64)

    print('model.evaluate', score)

    y_pred = model.predict_classes(x_test)

    print('roc auc score', metrics.roc_auc_score(y_test, y_pred))

    print('classification report:\n', classification_report(y_test, y_pred))
    print('number of ones in y_test', len([1 for x in y_test if x == 1]))
    print('number of (y > 0)s in y_pred', len([1 for x in y_pred if x > 0]))
    #print('number of (y[0] > 0)s in y_pred', len([1 for x in y_pred if x[0] > 0]))


    # BE advised!
    # model.add(Dense(2, activation='softmax'))
    #
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer=Adam(lr=learning_rate),
    #               metrics=['accuracy'])

    # saving model
    json_model = model.model.to_json()
    open('bureau_lstm.json', 'w').write(json_model)
    # saving weights
    model.model.save_weights('bureau_lstm_weights.h5', overwrite=True)

    return model


# nested k-fold training for the main (meta classifier) dataset creation
def nested_k_fold(create_dataset_filename):
    print('> nested_k_fold')
    # Load the created data from bureau_recurrent_data

    with open('bureau/bureau_targets.pickle', 'rb') as handle:
        bureau_targets = pickle.load(handle)

    with open('bureau/bureau_dataset_no_indices.pickle', 'rb') as handle:
        bureau_dataset = pickle.load(handle)

    with open('bureau/bureau_indices_list.pickle', 'rb') as handle:
        bureau_indices_list = pickle.load(handle)

    # Remove nans from bureau_targets (and coresponding indices from other lists)
    df = pd.DataFrame(columns=['bureau_targets'], data=bureau_targets)
    df['bureau_dataset'] = pd.Series(bureau_dataset)
    df['bureau_indices_list'] = pd.Series(bureau_indices_list)
    print('dataset.shape before NaN removal:', df.shape)

    df = df[np.isfinite(df['bureau_targets'])]
    print('dataset.shape:', df.shape)
    print('dataset.columns:', df.columns)

    # Indices will be necessary for later integration into other datasets
    bureau_targets = df['bureau_targets'].values
    bureau_dataset = df['bureau_dataset'].values
    bureau_indices_list = df['bureau_indices_list'].values
    del df

    # find all possible categorical values in bureau_dataset and find the longest one
    all_possible_values = set()
    maxlen = 0
    for ex in bureau_dataset:
        all_possible_values.update(ex)
        if len(ex) > maxlen:
            maxlen = len(ex)

    # integer encode the categorical values of 'bureau_dataset'
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list(all_possible_values))
    print('all_possible_values, integer_encoded', all_possible_values, integer_encoded)
    print('maximum sample length', maxlen)

    for i in range(len(bureau_dataset)):
        bureau_dataset[i] = label_encoder.transform(bureau_dataset[i])


    # Changing the datatype to integer (from float)
    bureau_targets = bureau_targets.astype(int)

    # initial meta split
    META_x_train, META_x_test, META_y_train, META_y_test = train_test_split(bureau_dataset, bureau_targets,
                                                        test_size=0.2, random_state=42)

    META_x_train = sequence.pad_sequences(META_x_train, maxlen=maxlen)
    META_x_test = sequence.pad_sequences(META_x_test, maxlen=maxlen)
    print('x_train shape:', META_x_train.shape)
    print('x_test shape:', META_x_test.shape)

    # META_x_train = META_x_train.reshape(418812, 97, 1)
    # META_x_test = META_x_test.reshape(104703, 97, 1)
    # print('x_train shape:', META_x_train.shape)
    # print('x_test shape:', META_x_test.shape)

    learning_rate = 0.0001
    dropout_ratio = 0.2
    class_0_weight = 1
    class_1_weight = 11.5

    # weights for the training samples, used for weighting the loss function (during training only)
    sample_weights = np.array([class_0_weight if y == 0 else class_1_weight for y in META_y_train])

    model = Sequential()
    model.add(LSTM(64, input_shape=(97, 1)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, activation='softmax'))

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    kf = KFold(n_splits=5)

    # The k-fold learning loop
    PRED = []
    PRED_Y = []
    count = 0
    for train_index, test_index in kf.split(META_x_train):

        x_train, x_test = META_x_train[train_index], META_x_train[test_index]
        y_train, y_test = np.array(META_y_train)[train_index], np.array(META_y_train)[test_index]

        print('\n\n==== run ====', count)

        x_train = x_train.reshape(x_train.shape[0], 97, 1)
        x_test = x_test.reshape(x_test.shape[0], 97, 1)

        sample_weights = np.array([class_0_weight if y == 0 else class_1_weight for y in y_train])

        ##############################################################################################
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(lr=learning_rate),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=64, epochs=15,
                  validation_split=0.1,
                  verbose=1,
                  sample_weight=sample_weights,
                  callbacks=[earlystopper],
                  shuffle=True)

        score = model.evaluate(x_test, y_test, batch_size=64)

        print('model.evaluate', score)

        y_pred = model.predict_classes(x_test)

        print('roc auc score', metrics.roc_auc_score(y_test, y_pred))

        print('classification report:\n', classification_report(y_test, y_pred))
        print('number of ones in y_test', len([1 for x in y_test if x == 1]))
        print('number of (y > 0)s in y_pred', len([1 for x in y_pred if x > 0]))
        ##############################################################################################
        PR = model.predict(x_test)

        PR2 = model.predict_proba(x_test)

        for i in range(len(PR)):
            PRED.append(np.concatenate((PR[i], PR2[i])))

        PRED_Y = np.concatenate((PRED_Y, y_test))

        count += 1

    np_PRED = np.array(PRED)
    PRED_Y = PRED_Y.astype(int)

    np.savetxt("bureau_lstm_x_train.csv", np_PRED, delimiter=",")
    np.savetxt("bureau_lstm_y_train.csv", PRED_Y, delimiter=",")

    TEST = []
    TEST_Y = []
    count = 0

    META_x_train = META_x_train.reshape(META_x_train.shape[0], 97, 1)
    META_x_test = META_x_test.reshape(META_x_test.shape[0], 97, 1)
    sample_weights = np.array([class_0_weight if y == 0 else class_1_weight for y in META_y_train])

    ##############################################################################################
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(META_x_train, META_y_train, batch_size=64, epochs=15,
              validation_split=0.1,
              verbose=1,
              sample_weight=sample_weights,
              callbacks=[earlystopper],
              shuffle=True)

    score = model.evaluate(META_x_test, META_y_test, batch_size=64)

    print('model.evaluate', score)

    y_pred = model.predict_classes(META_x_test)

    print('roc auc score', metrics.roc_auc_score(META_y_test, y_pred))

    print('classification report:\n', classification_report(META_y_test, y_pred))
    print('number of ones in y_test', len([1 for x in META_y_test if x == 1]))
    print('number of (y > 0)s in y_pred', len([1 for x in y_pred if x > 0]))
    ##############################################################################################
    TE = model.predict(META_x_test)

    TE2 = model.predict_proba(META_x_test)

    for i in range(len(TE2)):
        TEST.append(np.concatenate((TE[i], TE2[i])))

    TEST_Y = np.concatenate((TEST_Y, META_y_test))
    count += 1

    np_TEST = np.array(TEST)
    TEST_Y = TEST_Y.astype(int)

    np.savetxt("bureau_lstm_x_test.csv", np_TEST, delimiter=",")
    np.savetxt("bureau_lstm_y_test.csv", TEST_Y, delimiter=",")


if __name__ == "__main__":
    os.chdir(DIR_JURE)

    # df_app_train, df_app_test = baseline.init()
    # bureau_recurrent_data_init(df_app_train)
    # bureau_recurrent_data_preparation()
    # bureau_recurrent_model()

    # model = bureau_recurrent_model_no_embedding()
    nested_k_fold('bureau_lstm_dataset.csv')

    exit(0)
