import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn import metrics
import gc
import warnings


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
target = 'target'
predictors = train_df.columns.values.tolist()[2:]
NUM_FOLDS = 10
bayesian_tr_index, bayesian_val_index = list(StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1).split(train_df, train_df.target.values))[0]
NUM_ROUNDS = 1000
EARLY_STOPPING = 100
init_points = 100
n_iter = 100


def LGB_bayesian(
        bagging_fraction,
        bagging_freq,
        drop_rate,
        feature_fraction,
        lambda_l1,
        lambda_l2,
        learning_rate,
        max_bin,
        max_depth,
        min_data_in_leaf,
        min_gain_to_split,
        min_sum_hessian_in_leaf,
        num_leaves,
        skip_drop):
    """
    Used for bayesian optimization
    :param bagging_fraction:
    :param bagging_freq:
    :param drop_rate:
    :param feature_fraction:
    :param lambda_l1:
    :param lambda_l2:
    :param learning_rate:
    :param max_bin:
    :param max_depth:
    :param min_data_in_leaf:
    :param min_gain_to_split:
    :param min_sum_hessian_in_leaf:
    :param num_leaves:
    :param skip_drop:
    :return:
    """

    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    max_bin = int(max_bin)
    bagging_freq = int(bagging_freq)

    param = {
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'boosting': 'dart',
        'device_type': 'gpu',
        'drop_rate': drop_rate,
        'feature_fraction': feature_fraction,
        'gpu_device_id': 0,
        'gpu_platform_id': 0,
        'gpu_use_dp': True,
        'is_unbalance': True,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'learning_rate': learning_rate,
        'max_bin': max_bin,
        'max_depth': max_depth,
        'metric': 'auc',
        'min_data_in_leaf': min_data_in_leaf,
        'min_gain_to_split': min_gain_to_split,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'num_leaves': num_leaves,
        'num_threads': 4,
        'objective': 'binary',
        'save_binary': True,
        'skip_drop': skip_drop,
        'tree_learner': 'feature',
        'verbosity': 1,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
    }

    xg_train = lgb.Dataset(train_df.iloc[bayesian_tr_index][predictors].values,
                           label=train_df.iloc[bayesian_tr_index][target].values,
                           feature_name=predictors,
                           free_raw_data=False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[bayesian_val_index][predictors].values,
                           label=train_df.iloc[bayesian_val_index][target].values,
                           feature_name=predictors,
                           free_raw_data=False
                           )

    clf = lgb.train(param, xg_train, NUM_ROUNDS, valid_sets=[xg_valid], verbose_eval=250, early_stopping_rounds=EARLY_STOPPING)

    predictions = clf.predict(train_df.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)

    score = metrics.roc_auc_score(train_df.iloc[bayesian_val_index][target].values, predictions)

    return score

# Bounded region of parameter space
bounds_LGB = {
    'bagging_fraction': (0.0, 1.0),
    'bagging_freq': (0, 20),
    'drop_rate': (0.0, 1.0),
    'feature_fraction': (0.01, 1),
    'lambda_l1': (0, 200),
    'lambda_l2': (0, 200),
    'learning_rate': (0.0025, 0.3),
    'max_bin': (7, 1023),
    'max_depth': (1, 100),
    'metric': 'auc',
    'min_data_in_leaf': (0, 200),
    'min_gain_to_split': (0, 2.0),
    'min_sum_hessian_in_leaf': (0, 150),
    'num_leaves': (1, 100),
    'skip_drop': (0.0, 1.0),
}

from bayes_opt import BayesianOptimization

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


print(LGB_BO.max['target'])

print(LGB_BO.max['params'])

param_lgb = {
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),
    'boosting': 'dart',
    'device_type': 'gpu',
    'drop_rate': LGB_BO.max['params']['drop_rate'],
    'feature_fraction': LGB_BO.max['params']['feature_fraction'],
    'gpu_device_id': 0,
    'gpu_platform_id': 0,
    'gpu_use_dp': True,
    'is_unbalance': True,
    'lambda_l1': LGB_BO.max['params']['lambda_l1'],
    'lambda_l2': LGB_BO.max['params']['lambda_l2'],
    'learning_rate': LGB_BO.max['params']['learning_rate'],
    'max_bin': int(LGB_BO.max['params']['max_bin']),
    'max_depth': int(LGB_BO.max['params']['max_depth']),
    'metric': 'auc',
    'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
    'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
    'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
    'num_leaves': int(LGB_BO.max['params']['num_leaves']),
    'num_threads': 4,
    'objective': 'binary',
    'save_binary': True,
    'skip_drop': LGB_BO.max['params']['skip_drop'],
    'tree_learner': 'feature',
    'verbosity': 1,
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    }

nfold = 10
gc.collect()

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))
predictions = np.zeros((len(test_df), nfold))

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data=False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data=False
                           )

    clf = lgb.train(param_lgb, xg_train, NUM_ROUNDS, valid_sets=[xg_valid], verbose_eval=250, early_stopping_rounds=EARLY_STOPPING)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration)

    predictions[:, i - 1] += clf.predict(test_df[predictors], num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))

print("Rank averaging on", nfold, "fold predictions")
rank_predictions = np.zeros((predictions.shape[0],1))
for i in range(nfold):
    rank_predictions[:, 0] = np.add(rank_predictions[:, 0], rankdata(predictions[:, i].reshape(-1,1))/rank_predictions.shape[0])

rank_predictions /= nfold

sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = rank_predictions

sub_df.to_csv("results/submission18.csv", index=False)
