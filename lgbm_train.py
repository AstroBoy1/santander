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
NUM_ROUNDS = 10000
EARLY_STOPPING = 100


def LGB_bayesian(
        bagging_fraction,
        bagging_freq,
        boosting,
        device_type,
        drop_rate,
        feature_fraction,
        gpu_device_id,
        gpu_platform_id,
        gpu_use_dp,
        is_unbalance,
        lambda_l1,
        lambda_l2,
        learning_rate,
        max_bin,
        max_delta,
        max_depth,
        metric,
        min_data_in_leaf,
        min_gain_to_split,
        min_sum_hessian_in_leaf,
        num_iterations,
        num_leaves,
        num_threads,
        objective,
        save_binary,
        skip_drop,
        tree_learner,
        uniform_drop,
        verbosity):

    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    max_bin = int(max_bin)
    num_threads = int(num_threads)

    param = {
        'num_leaves': num_leaves,
        'max_bin': 119,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 0.4,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id' : 0,
        'boosting' : 'dart'
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
    'num_leaves': (5, 100),
    'min_data_in_leaf': (5, 100),
    'learning_rate': (0.01, 0.3),
    'min_sum_hessian_in_leaf': (0.00001, 0.01),
    'feature_fraction': (0.05, 0.5),
    'lambda_l1': (0, 5.0),
    'lambda_l2': (0, 5.0),
    'min_gain_to_split': (0, 1.0),
    'max_depth':(3,15),
}

from bayes_opt import BayesianOptimization

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

init_points = 1000
n_iter = 1000

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


print(LGB_BO.max['target'])

print(LGB_BO.max['params'])

param_lgb = {
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), # remember to int here
        'max_bin': int(LGB_BO.max['params']['num_leaves']),
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), # remember to int here
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
        'bagging_fraction': 0.4,
        'bagging_freq': 5,
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'lambda_l1': LGB_BO.max['params']['lambda_l1'],
        'lambda_l2': LGB_BO.max['params']['lambda_l2'],
        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), # remember to int here
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
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
