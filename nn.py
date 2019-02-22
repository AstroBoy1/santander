from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import svm
from collections import Counter


if __name__ == "__main__":

    length = 10000
    train_data_df = pd.read_csv("data/train.csv")
    train_data_x = train_data_df.drop(columns=["ID_code", "target"]).values
    train_data_y = train_data_df["target"].values
    valid_data_df = pd.read_csv("data/test.csv")
    valid_data_x = valid_data_df.drop(columns=["ID_code"]).values

    clf = svm.SVC()
    print("Fitting data")
    clf.fit(train_data_x[:length], train_data_y[:length])
    print("Predicting data")
    valid_pred = clf.predict(valid_data_x[:length])
    print("Scoring data")
    output_df = pd.DataFrame()
    output_df["ID_code"] = valid_data_df["ID_code"]
    output_df["target"] = valid_pred

    #score = roc_auc_score(valid_data_y[:length], valid_pred[:length])
    #print(score)

    # model = Sequential()
    # model.add(Dense(units=64, activation='relu', input_dim=100))
    # model.add(Dense(units=10, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=5, batch_size=32)
    # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    # classes = model.predict(x_test, batch_size=128)