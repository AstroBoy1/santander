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
from keras.optimizers import SGD
# LOAD DATA
df_train = pd.read_csv('data/train.csv', index_col=0)
y_train = df_train.pop('target')
len_train = len(df_train)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_all = pd.concat((df_train, df_test), sort=False)
prev_cols = df_all.columns

# PREPROCESS
scaler = StandardScaler()
df_all[prev_cols] = scaler.fit_transform(df_all[prev_cols])
df_train = df_all[0:len_train]
df_test = df_all[len_train:]

# CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# LOGGER
class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=100, lr_patience=20, out_fn='', log_fn=''):
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
    l1 = Dense(128, activation='relu')(inp)
    d1 = Dropout(0.5)(l1)
    l2 = Dense(64, activation='relu')(d1)
    d2 = Dropout(0.5)(l2)
    l3 = Dense(16, activation='relu')(d2)
    d3 = Dropout(0.5)(l3)
    fl = Flatten()(d3)
    preds = Dense(1, activation='sigmoid')(fl)
    model = Model(inputs=inp, outputs=preds)
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model


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
    logger = Logger(patience=100, out_path='./', out_fn='cv_{}.h5'.format(c))
    model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=1000, verbose=2, batch_size=256,
              callbacks=[logger])
    model.load_weights('cv_{}.h5'.format(c))
    X_test = np.reshape(df_test.values, (200000, 200, 1))
    curr_preds = model.predict(X_test, batch_size=2048)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
    break
#auc = roc_auc_score(y_train, oof_preds)
#print("CV_AUC: {}".format(auc))

# SAVE DATA
#preds = np.asarray(preds)
#preds = preds.reshape((5, 200000))
#preds_final = np.mean(preds.T, axis=1)
submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = preds[0]
submission.to_csv('final_submission.csv', index=False)

#for layer in model.layers:
#    print(layer.output_shape)
