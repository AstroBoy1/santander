import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
np.random.seed(0)
np.set_printoptions(suppress=True)
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import keras.backend as K
import warnings
warnings.simplefilter("ignore")
from keras.layers.recurrent import LSTM


if __name__ == "__main__":
    train_fn = 'data/train.csv'
    valid_fn = 'data/test.csv'
    pred_fn = 'lstm_preds.csv'
    train_data_df = pd.read_csv(train_fn)
    test_data_df = pd.read_csv(valid_fn)
    train_data_x = train_data_df.drop(columns=["ID_code", "target"]).values
    train_data_y = train_data_df["target"].values
    test_data_x = test_data_df.drop(columns=["ID_code"]).values

    model = Sequential()
    model.add(
        LSTM(64, input_shape=(200, 1), dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True))
    model.add(
        LSTM(16, input_shape=(200, 1), dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    num_samples = 200000
    x = train_data_x[:num_samples].reshape(num_samples, 200, 1)
    y = train_data_y[:num_samples]


    # LOGGER
    class Logger(callbacks.Callback):
        def __init__(self, out_path="/", patience=10, lr_patience=3, out_fn='', log_fn=''):
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
                    K.set_value(self.model.optimizer.lr, 0.75 * lr)
                    print("Setting lr to {}".format(0.75 * lr))
                    self.no_improve_lr = 0
            return


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    preds = []
    # oof_preds = np.zeros((num_samples, 1))
    c = 0
    for train, valid in cv.split(x, y):
        c += 1
        logger = Logger(patience=10, out_path="/", out_fn='cv_{}.h5'.format(c))
        model.fit(x[train][:], y[train][:], nb_epoch=5, validation_data=(x[valid][:], y[valid][:]), callbacks=[logger])
        model.load_weights("/" + 'cv_{}.h5'.format(c))
        X_test = np.reshape(test_data_x, (200000, 200, 1))
        curr_preds = model.predict(X_test, batch_size=2048)
        # oof_preds[valid] = model.predict(x[valid])
        preds.append(curr_preds)

    # auc = roc_auc_score(y_train, oof_preds)
    # print("CV_AUC: {}".format(auc))

    # SAVE DATA
    preds = np.asarray(preds)
    preds = preds.reshape((5, 200000))
    preds_final = np.mean(preds.T, axis=1)
    submission = pd.read_csv('data/sample_submission.csv')
    submission['target'] = preds_final
    submission.to_csv('lstm_submission.csv', index=False)