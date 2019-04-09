# IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, GaussianNoise
from keras import callbacks
import keras.backend as K
import pickle
from keras import optimizers
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from __future__ import print_function
from keras.layers.core import Dense, Dropout, Activation

# LOAD DATA
df_train = pd.read_csv('data/train.csv', index_col=0)
num_samples = len(df_train)
#num_samples = 10000
df_train = df_train[:num_samples]
y_train = df_train.pop('target')
len_train = len(df_train)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_all = pd.concat((df_train, df_test), sort=False)
prev_cols = df_all.columns
num_epochs = 1000
patience_epochs = 50

# PREPROCESS
scaler = StandardScaler()
df_all[prev_cols] = scaler.fit_transform(df_all[prev_cols])
df_train = df_all[0:len_train]
df_test = df_all[len_train:]

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    df_train = pd.read_csv('data/train.csv', index_col=0)
    (x_train, y_train), (x_test, y_test) =
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

# LOGGER
class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=10, lr_patience=3, out_fn='', log_fn=''):
        self.auc = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
        cv_true = self.validation_data[1]
        auc_val = roc_auc_score(cv_true, cv_pred)
        if self.auc < auc_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
            self.auc = auc_val
            self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, 0.75*lr)
                print("Setting lr to {}".format(0.75*lr))
                self.no_improve_lr = 0

        return

# MODEL DEF
def _Model():
    inp = Input(shape=(200, 1))
    d1 = Dense(128, activation='relu')(inp)
    d1 = Dropout(rate=0.5)(d1)
    d1 = Dense(64, activation='relu')(d1)
    d1 = Dropout(rate=0.5)(d1)
    d1 = Dense(16, activation='relu')(d1)
    fl = Flatten()(d1)
    preds = Dense(1, activation='sigmoid')(fl)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(200,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
#    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    #RUN
    preds = []
    c = 0
    oof_preds = np.zeros((len(df_train), 1))
    for train, valid in cv.split(df_train, y_train):
        print("VAL %s" % c)
        X_train = np.reshape(df_train.iloc[train].values, (-1, 200, 1))
        y_train_ = y_train.iloc[train].values
        X_valid = np.reshape(df_train.iloc[valid].values, (-1, 200, 1))
        y_valid = y_train.iloc[valid].values
        model = _Model()
        logger = Logger(patience=patience_epochs, out_path='./', out_fn='cv_{}.h5'.format(c))
        history = model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=num_epochs, verbose=1, batch_size=256,
                  callbacks=[logger])
        model.load_weights('cv_{}.h5'.format(c))
        X_test = np.reshape(df_test.values, (200000, 200, 1))
        curr_preds = model.predict(X_test, batch_size=2048)
        oof_preds[valid] = model.predict(X_valid)
        preds.append(curr_preds)
        c += 1
        history_fn = "history" + str(c)
        with open(history_fn, 'wb') as fn:
            pickle.dump(history.history, fn)
    auc = roc_auc_score(y_train, oof_preds)
    print("CV_AUC: {}".format(auc))

    # SAVE DATA
    preds = np.asarray(preds)
    preds = preds.reshape((5, 200000))
    preds_final = np.mean(preds.T, axis=1)
    #submission = pd.read_csv('./../input/sample_submission.csv')
    submission = pd.DataFrame()
    submission = pd.read_csv("data/sample_submission.csv")
    submission['target'] = preds_final
    submission.to_csv('results/submission20.csv', index=False)
