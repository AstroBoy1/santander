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


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    df_train = pd.read_csv('data/train.csv', index_col=0)[:1000]
    y = df_train['target'].values
    x = df_train.iloc[:, df_train.columns != 'target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test


# class Logger(callbacks.Callback):
#     def __init__(self, out_path='./', patience=10, lr_patience=3, out_fn='', log_fn=''):
#         self.auc = 0
#         self.path = out_path
#         self.fn = out_fn
#         self.patience = patience
#         self.lr_patience = lr_patience
#         self.no_improve = 0
#         self.no_improve_lr = 0
#
#     def on_train_begin(self, logs={}):
#         return
#
#     def on_train_end(self, logs={}):
#         return
#
#     def on_epoch_begin(self, epoch, logs={}):
#         return
#
#     def on_batch_begin(self, batch, logs={}):
#         return
#
#     def on_batch_end(self, batch, logs={}):
#         return
#
#     def on_epoch_end(self, epoch, logs={}):
#         cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
#         cv_true = self.validation_data[1]
#         auc_val = roc_auc_score(cv_true, cv_pred)
#         if self.auc < auc_val:
#             self.no_improve = 0
#             self.no_improve_lr = 0
#             print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
#             self.auc = auc_val
#             self.model.save(self.path + self.fn, overwrite=True)
#         else:
#             self.no_improve += 1
#             self.no_improve_lr += 1
#             print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
#             if self.no_improve >= self.patience:
#                 self.model.stop_training = True
#             if self.no_improve_lr >= self.lr_patience:
#                 lr = float(K.get_value(self.model.optimizer.lr))
#                 K.set_value(self.model.optimizer.lr, 0.75*lr)
#                 print("Setting lr to {}".format(0.75*lr))
#                 self.no_improve_lr = 0
#
#         return


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
    model.add(Dense(512, input_shape=(784,)))
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
    model.add(Activation('sigmoid'))

    #logger = Logger(patience=patience_epochs, out_path='./', out_fn='cv_{}.h5'.format(c))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['roc_auc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    patience_epochs = 50
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
