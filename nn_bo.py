from __future__ import print_function
import numpy as np
from keras import callbacks
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from hyperas import optim
from hyperas.distributions import choice, uniform
import pandas as pd
from sklearn.model_selection import train_test_split
import keras.backend as K
from bayes_opt import BayesianOptimization
from functools import partial
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.layers import Flatten

class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=10, lr_patience=3, out_fn='', log_fn='', decay=0.5):
        self.auc = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0
        self.losses = []
        self.auc_list = []
        self.decay = decay

    def on_train_begin(self, logs={}):
        self.auc_list = []
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
        cv_pred = self.model.predict(self.validation_data[0])
        cv_true = self.validation_data[1]
        auc_val = roc_auc_score(cv_true, cv_pred)
        self.auc_list.append(auc_val)
        if self.auc < auc_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
            self.auc = auc_val
            #self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, self.decay*lr)
                print("Setting lr to {}".format(self.decay*lr))
                self.no_improve_lr = 0
        return


def get_model(l1, drop1, l2, drop2):
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
    model.add(Dense(int(l1), input_shape=(200, 1), name="l1"))
    model.add(Activation('relu'))
    model.add(Dropout(drop1, name="drop1"))
    model.add(Dense(int(l2), name="l2"))
    model.add(Activation('relu'))
    model.add(Dropout(drop2, name="drop2"))
    model.add(Flatten())
    model.add(Dense(1, name="l3"))
    model.add(Activation('sigmoid'))
    return model


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def fit_with(l1, drop1, l2, drop2, momentum, decay, batch_size, lr):
    c = 0
    # Create the model using a specified hyperparameters.
    model = get_model(l1, drop1, l2, drop2)

    # Train the model for a specified number of epochs.
    logger = Logger(patience=patience_epochs, lr_patience=lr_patience, out_path='./', out_fn='cv_{}.h5'.format(c))
    optimizer = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', metrics=[auc],  optimizer=optimizer)
    best_scores = []
    for train, valid in cv.split(x, y):
        x_train = np.reshape(x[train][:], (-1, 200, 1))
        y_train = y[train][:]
        x_valid = np.reshape(x[valid][:], (-1, 200, 1))
        y_valid = y[valid][:]
    # Train the model with the train dataset.
        model.fit(x_train, y_train, epochs=1000, batch_size=int(batch_size), validation_data=[x_valid, y_valid], callbacks=[logger])
        best_score = max(logger.auc_list)
        best_scores.append(best_score)
        logger.auc_list = []
    # Evaluate the model with the eval dataset.
    #score = model.evaluate(eval_ds, steps=10, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    # Return the accuracy.
    # best_score = max(logger.auc_list)
    print("AUCs: ", best_scores)
    fh.write(str(best_scores))
    fh.write("l1: %d drop1: %d l2: %d drop2: %d momentum: %d decay: %d batch_size: %d lr: %d" % (l1, drop1, l2, drop2, momentum, decay, batch_size, lr))
    fh.write(str(np.mean(best_scores)))
    fh.write("\n")
    return np.mean(best_scores)


if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv', index_col=0)[:200000]
    y = df_train['target'].values
    x = df_train.iloc[:, df_train.columns != 'target'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    #x = StandardScaler().fit_transform(x)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # l1, act1, drop1, l2, act2, drop2
    # Bounded region of parameter space
    patience_epochs = 10
    lr_patience = 3
    fh = open("nn_logs.txt", "a")
    pbounds = {'l1': (50, 100), 'drop1': (0, 0.5), 'l2': (10, 25), 'drop2': (0, 0.5), 'momentum': (0, 1), 'decay': (0.1, 0.75), 'batch_size': (1, 256), 'lr': (1e-3, 5e-2)}
    fit_with_partial = partial(fit_with)
    print("Creating optimizer")
    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    print("Optimizing")
    optimizer.maximize(init_points=10, n_iter=10, acq="ucb")

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)
    fh.write(optimizer.max)
    fh.close()
