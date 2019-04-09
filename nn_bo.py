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


class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=10, lr_patience=3, out_fn='', log_fn=''):
        self.auc = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0
        self.losses = []
        self.auc_list = []

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
        self.auc_list.append(auc_val)
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


def get_model():
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
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def fit_with():
    c = 0
    # Create the model using a specified hyperparameters.
    model = get_model()

    # Train the model for a specified number of epochs.
    logger = Logger(patience=patience_epochs, out_path='./', out_fn='cv_{}.h5'.format(c))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, callbacks=[logger])
    # Train the model with the train dataset.
    history = model.fit(x, y, epochs=1, batch_size=64, validation_split=0.2)

    # Evaluate the model with the eval dataset.
    #score = model.evaluate(eval_ds, steps=10, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    # Return the accuracy.
    best_score = max(history.auc_list)
    print("Best AUC: ", best_score)
    return best_score


if __name__ == '__main__':
    patience_epochs = 50
    df_train = pd.read_csv('data/train.csv', index_col=0)[:1000]
    y = df_train['target'].values
    x = df_train.iloc[:, df_train.columns != 'target'].values
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Bounded region of parameter space
    pbounds = {'dropout2_rate': (0.1, 0.5), 'lr': (1e-4, 1e-2)}
    fit_with_partial = partial(fit_with)
    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=10, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)
