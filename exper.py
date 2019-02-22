import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import svm
from collections import Counter

if __name__ == "__main__":
    # length = 1000
    # print("Reading data")
    # train_data_df = pd.read_csv("data/train.csv")
    # train_data_x = train_data_df.drop(columns=["ID_code", "target"]).values
    # train_data_y = train_data_df["target"].values
    # clf = svm.SVC()
    # print("Fitting data")
    # clf.fit(train_data_x[:length], train_data_y[:length])
    # # output_df.to_csv("submission.csv", index=False)
    # print("Predicting data")
    # train_pred = clf.predict(train_data_x[:length])
    # print("Scoring data")
    # test = [1] * length
    # # score = roc_auc_score(train_data_y[:length], train_pred[:length])
    # score = roc_auc_score(train_data_y[:length], test)
    # print(score)

    df = pd.read_csv("data/sample_submission.csv")
    #c = Counter(df['target'])
    #print(c)
    choices = np.random.choice(200000, 20098, replace=False)
    targets = [0] * 200000
    for index in choices:
        targets[index] = 1
    df['target'] = targets
    df.to_csv("random.csv", index=False)
