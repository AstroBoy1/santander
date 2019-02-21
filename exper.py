import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import svm

if __name__ == "__main__":
    length = 1000
    print("Reading data")
    train_data_df = pd.read_csv("data/train.csv")
    train_data_x = train_data_df.drop(columns=["ID_code", "target"]).values
    train_data_y = train_data_df["target"].values
    clf = svm.SVC()
    print("Fitting data")
    clf.fit(train_data_x[:length], train_data_y[:length])
    # output_df.to_csv("submission.csv", index=False)
    print("Predicting data")
    train_pred = clf.predict(train_data_x[:length])
    print("Scoring data")
    test = [1] * length
    # score = roc_auc_score(train_data_y[:length], train_pred[:length])
    score = roc_auc_score(train_data_y[:length], test)
    print(score)
