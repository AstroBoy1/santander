import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from vecstack import StackingTransformer
import pandas as pd
np.random.seed(0)
np.set_printoptions(suppress=True)
# Models
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Stacking
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
import warnings
warnings.simplefilter("ignore")
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
import random
from sklearn.metrics import roc_auc_score
import joblib
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

train_fn = '/content/gdrive/My Drive/santander_data/train.csv'
valid_fn = '/content/gdrive/My Drive/santander_data/test.csv'
pred_fn = '/content/gdrive/My Drive/santander_data/submission.csv'
train_data_df = pd.read_csv(train_fn)
test_data_df = pd.read_csv(valid_fn)
train_data_x = train_data_df.drop(columns=["ID_code", "target"]).values
train_data_y = train_data_df["target"].values
test_data_x = test_data_df.drop(columns=["ID_code"]).values

n_classes = 2
length = 1000
save_directory = '/content/gdrive/My Drive/santander_results/'
early_stopping_rounds = 10
# How many times to sample the features
num_subsets = 5
X, y = train_data_x[:length], train_data_y[:length]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgb_params = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1,
    'num_iterations': 100000
}

xgb_params = {
    'tree_method': 'hist',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.0936165921314771,
    'max_depth': 2,
    'colsample_bytree': 0.3561271102144279,
    'subsample': 0.8246604621518232,
    'min_child_weight': 53,
    'gamma': 9.943467991283027,
    'silent': 1,
    'n_estimators': 999999
}

catboost_params = {
    'subsample': 0.36,
    'loss_function': 'Logloss',
    'random_strength': 0,
    'max_depth': 3,
    'eval_metric': "AUC",
    'learning_rate': 0.02,
    'iterations': 100000,
    'bootstrap_type': 'Bernoulli',
    'l2_leaf_reg': 0.3,
    'task_type': "GPU",
    'random_seed': 432013,
    'od_type': "Iter",
    'border_count': 128
}

cbp1 = {
    "iterations": 10000,
    "thread_count": 4,
    "loss_function": 'Logloss',
    "eval_metric": 'AUC',
    "random_seed": 56498323,
    "depth": 2,  # default 6
    "learning_rate": 0.04,
    "l2_leaf_reg": 2,  # default 3
    "verbose": 1,
    "use_best_model": True,
    "od_type": 'Iter',
    "od_wait": 250,
    "nan_mode": "Min"
}

cbp2 = {
    "loss_function": "Logloss",
    "eval_metric": 'AUC',
    "task_type": "GPU",
    "learning_rate": 0.01,
    "iterations": 100000,
    "random_seed": 42,
    "od_type": "Iter",
    "depth": 10
}

cbp3 = {
    "subsample": 0.36,
    "random_strength": 0,
    "max_depth": 3,
    "loss_function": 'Logloss',
    "eval_metric": 'AUC', "learning_rate": 0.02,
    "iterations": 100000,
    "bootstrap_type": 'Bernoulli',
    "l2_leaf_reg": 0.3,
    "task_type": "GPU",
    "random_seed": 432013,
    "od_type": "Iter",
    "border_count": 128
}

cbp4 = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": "GPU",
    "learning_rate": 0.01,
    "iterations": 100000,
    "l2_leaf_reg": 50,
    "random_seed": 432013,
    "od_type": "Iter",
    "depth": 5,
    "border_count": 64
}

random_state = 432013
# https://www.kaggle.com/marcospcsj/kernel-cod-valid-cruzada-lgbm
lgbp1 = {
    'bagging_freq': 5, 'bagging_fraction': 0.335, 'boost_from_average': 'false', 'boost': 'gbdt',
    'feature_fraction': 0.041, 'learning_rate': 0.0083, 'max_depth': -1, 'metric': 'auc',
    'min_data_in_leaf': 80, 'min_sum_hessian_in_leaf': 10.0, 'num_leaves': 13, 'num_threads': 4,
    'tree_learner': 'serial', 'objective': 'binary', 'verbosity': 1, 'num_iterations': 100000
}

# https://www.kaggle.com/bejeweled/stats-features-lgbm
lgbp2 = {
    "objective": "binary",
    "metric": "auc",
    "boosting": 'gbdt',
    "max_depth": 90,
    "num_leaves": 15,
    "learning_rate": 0.01000123,
    "bagging_freq": 5,
    "bagging_fraction": 0.4,
    "feature_fraction": 0.05,
    "min_data_in_leaf": 150,
    "min_sum_heassian_in_leaf": 15,
    "tree_learner": "voting",
    "boost_from_average": "false",
    "lambda_l1": 10,
    "lambda_l2": 10,
    "bagging_seed": random_state,
    "verbosity": 1,
    "seed": random_state,
    'num_iterations': 100000
}

# https://www.kaggle.com/sandan0o0/lgb-regression
lgbp3 = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective': 'binary',
         'max_depth': -1,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 2019,
         'num_iterations': 100000}

# https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
lgbp4 = {
    "objective": "binary",
    "metric": "auc",
    "boosting": 'gbdt',
    "max_depth": -1,
    "num_leaves": 13,
    "learning_rate": 0.01,
    "bagging_freq": 5,
    "bagging_fraction": 0.4,
    "feature_fraction": 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    # "lambda_l1" : 5,
    # "lambda_l2" : 5,
    "bagging_seed": random_state,
    "verbosity": 1,
    "seed": random_state,
    'num_iterations': 100000
}

# https://www.kaggle.com/indranilbhattacharya/santander-xgb-lgb-starter
xgbp1 = {'objective': "binary:logistic",
         'eval_metric': "auc",
         'max_depth': 4,
         'eta': 0.05,
         'gamma': 5,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'min_child_weight': 50,
         'colsample_bylevel': 0.7,
         'lambda': 1,
         'alpha': 0,
         'booster': "gbtree",
         'silent': 0,
         'n_estimators': 999999

         }

# https://www.kaggle.com/marcospcsj/simple-xgboost-test
xgbp2 = {'max_depth': 3,
         'silent': 1,
         'eval_metric': 'auc',
         'eta': 0.28071497637474263,  # learning rate
         'gamma': 0,
         'min_child_weight': 0.2784483175645849,
         'objective': 'binary:logistic',
         'n_estimators': 999999

         }

# https://www.kaggle.com/bogorodvo/starter-code-saving-and-loading-lgb-xgb-cb
xgbp3 = {'max_depth': 2, 'eval_metric': 'auc',
         'n_estimators': 999999,
         'colsample_bytree': 0.3,
         'learning_rate': 0.02,
         'objective': 'binary:logistic',
         'n_jobs': -1
         }

# https://www.kaggle.com/silverstone1903/xgboost-baseline
xgbp4 = {'tree_method': 'hist',
         'objective': 'binary:logistic',
         'eval_metric': 'auc',
         'learning_rate': 0.0936165921314771,
         'max_depth': 2,
         'colsample_bytree': 0.3561271102144279,
         'subsample': 0.8246604621518232,
         'min_child_weight': 53,
         'gamma': 9.943467991283027,
         'silent': 1,
         'n_estimators': 999999
         }

xgbp_list = []
lgbp_list = []
cbp_list = []

xgbp_list.append(xgbp1)
xgbp_list.append(xgbp2)
xgbp_list.append(xgbp3)
xgbp_list.append(xgbp4)
xgbp_list.append(xgb_params)

lgbp_list.append(lgbp1)
lgbp_list.append(lgbp2)
lgbp_list.append(lgbp3)
lgbp_list.append(lgbp4)
lgbp_list.append(lgb_params)

cbp_list.append(cbp1)
cbp_list.append(cbp2)
cbp_list.append(cbp3)
cbp_list.append(cbp4)
cbp_list.append(catboost_params)


def auc(y_true, y_pred):
    """ROC AUC metric for both binary and multiclass classification.

    Parameters
    ----------
    y_true : 1d numpy array
        True class labels
    y_pred : 2d numpy array
        Predicted probabilities for each class
    """
    ohe = OneHotEncoder(sparse=False)
    y_true = ohe.fit_transform(y_true.reshape(-1, 1))
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score


class WrapLGB(LGBMClassifier):
    """This is template for user-defined class wrapper.
    Use this template to pass any ``fit`` and ``predict`` arguments.
    """

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
        return super(WrapLGB, self).fit(X_tr, y_tr,
                                        early_stopping_rounds=early_stopping_rounds,
                                        eval_set=[(X_val, y_val)],
                                        eval_metric='auc', verbose=1)

    def predict(self, X):
        return super(WrapLGB, self).predict(X,
                                            num_iteration=self.best_iteration_)


class WrapXGB(XGBClassifier):
    """This is template for user-defined class wrapper.
    Use this template to pass any ``fit`` and ``predict`` arguments.
    """

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
        return super(WrapXGB, self).fit(X_tr, y_tr,
                                        early_stopping_rounds=early_stopping_rounds,
                                        eval_set=[(X_val, y_val)],
                                        eval_metric='auc', verbose=1)

    def predict(self, X):
        return super(WrapXGB, self).predict(X,
                                            num_iteration=self.best_iteration_)


class WrapCB(CatBoostClassifier):
    """This is template for user-defined class wrapper.
    Use this template to pass any ``fit`` and ``predict`` arguments.
    """

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
        return super(WrapCB, self).fit(X_tr, y_tr,
                                       early_stopping_rounds=early_stopping_rounds,
                                       eval_set=[(X_val, y_val)], verbose=1)

    def predict(self, X):
        return super(WrapCB, self).predict(X,
                                           num_iteration=self.best_iteration_)

# Specify steps of Pipeline

# Same paramaters to models, but different subfeatures by:
# 1. Select a random number for the number of features to select
# 2. Randomly select that many features


subsets = []
features = list(range(200))
number = list(range(1, 201))
pipe_models_1 = []
for num in range(num_subsets):
    num_features = np.random.choice(number, size=1)[0]
    subset = np.random.choice(features, size=num_features, replace=False, p=None)
    subsets.append(subset)
    xgb_params_rand = random.choice(xgbp_list)
    lgb_params_rand = random.choice(lgbp_list)
    catboost_params_rand = random.choice(cbp_list)
    print("xgb params random chosen: ", xgb_params_rand)
    print("lgb params random chosen: ", lgb_params_rand)
    print("catboost params random chosen: ", catboost_params_rand)

    cl1 = ('gnb' + str(num), make_pipeline(ColumnSelector(cols=tuple(subset)), GaussianNB()))
    cl2 = ('xgb' + str(num), make_pipeline(ColumnSelector(cols=tuple(subset)), WrapXGB(**xgb_params_rand)))
    cl3 = ('lgbm' + str(num), make_pipeline(ColumnSelector(cols=tuple(subset)), WrapLGB(**lgb_params_rand)))
    cl4 = ('cb' + str(num), make_pipeline(ColumnSelector(cols=tuple(subset)), WrapCB(**catboost_params_rand)))
    pipe_models_1.append(cl1)
    pipe_models_1.append(cl2)
    pipe_models_1.append(cl3)
    pipe_models_1.append(cl4)
print("Number of level 1 models: ", len(pipe_models_1))
print("Number of subsets: ", len(subsets))
print("Subsets: ", subsets)

subset = list(range(0, len(pipe_models_1), 2))
xgb_params_rand = random.choice(xgbp_list)
lgb_params_rand = random.choice(lgbp_list)
cb_params_rand = random.choice(cbp_list)
print("xgb params random chosen: ", xgb_params_rand)
print("lgb params random chosen: ", lgb_params_rand)
print("catboost params random chosen: ", cb_params_rand)
pipe_models_2 = [
    ('xgb', make_pipeline(ColumnSelector(cols=tuple(subset)), WrapXGB(**xgb_params_rand))),
    ('lgbm', make_pipeline(ColumnSelector(cols=tuple(subset)), WrapLGB(**lgb_params_rand))),
    ('cb', make_pipeline(ColumnSelector(cols=tuple(subset)), WrapCB(**cb_params_rand)))
]

stack1 = StackingTransformer(pipe_models_1,
                             regression=False,
                             variant='A',
                             needs_proba=True,
                             metric=auc,
                             n_folds=5,
                             stratified=True,
                             shuffle=True,
                             random_state=0,
                             verbose=1)

stack2 = StackingTransformer(pipe_models_2,
                             regression=False,
                             variant='A',
                             needs_proba=True,
                             metric=auc,
                             n_folds=5,
                             stratified=True,
                             shuffle=True,
                             random_state=0,
                             verbose=1)


class Rank(BaseEstimator, TransformerMixin):
    """Define fit and transform for sklearn pipeline"""
    def __init__(self, method='average'):
        self.method = method

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.apply_along_axis(rankdata, 0, X)


stack3 = make_pipeline(Rank(), WrapCB(**catboost_params))

steps = [('stack1', stack1),
         ('stack2', stack2),
         ('stack3', stack3)]

# Init Pipeline
pipe = Pipeline(steps)

# Fit
pipe = pipe.fit(X_train, y_train)

y_pred = pipe.predict_proba(X_test)

# Final prediction score
# print('Final prediction score: %.8f' % log_loss(y_test, y_pred))
y_pred_final = [elem[1] for elem in y_pred]
roc_auc_score(y_test, y_pred_final)

y_pred = pipe.predict_proba(train_data_x)
y_pred_final = [elem[1] for elem in y_pred]
roc_auc_score(train_data_y, y_pred_final)

# Save Pipeline
_ = joblib.dump(pipe, save_directory + 'pipe_with_stack.pkl')
# Load Pipeline
pipe_loaded = joblib.load(save_directory + 'pipe_with_stack.pkl')

kaggle_pred = pipe.predict_proba(test_data_x)

output_df = pd.DataFrame()
output_df["ID_code"] = test_data_df["ID_code"]
pred_final = [elem[1] for elem in kaggle_pred]
output_df["target"] = pred_final
output_df.to_csv(save_directory + "predictions.csv", index=False)
