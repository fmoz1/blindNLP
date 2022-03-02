# conditional search spaces with hyperopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from hyperopt import hp, fmin, Trials
from hyperopt import anneal
from sympy import hyperexpand

from tutorial.smac_mnist import X

# determine the hyperparameter space
# model is a hyperparameter

# the nested space
# note each algo's hyperparams is given a unique name, e.g., *_rf, *_gbm
param_grid = hp.choice('classifier', [
    # algo 1
    {'model': LogisticRegression,
     'params': {
         'penalty': hp.choice('penalty', ['l1', 'l2']),
         'C': hp.uniform('C', 0.001, 10),
         'solver': 'saga',  # the only solver that works with both penalties
     }},
    # algo 2
    {'model': RandomForestClassifier,
     'params': {
         'n_estimators': hp.quniform('n_estimators_rf', 50, 1500, 50),
         'max_depth': hp.quniform('max_depth_rf', 1, 5, 1),
         'criterion': hp.choice('criterion_rf', ['gini', 'entropy']),
     }},

    # algo 3
    {'model': GradientBoostingClassifier,
     'params':
     {'n_estimators': hp.quniform('n_estimators_gbm', 50, 1500, 50),
      'max_depth': hp.quniform('max_depth_gbm', 1, 5, 1),
      'criterion': hp.choice('criterion_gbm', ['mse', 'friedman_mse']),
      }}
])

# define the objective function


def objective(params):
    # initiate the model
    model = params['model']()

    hyperparams = params['params']

    try:
        # not exist if sampling from logit
        hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
        hyperparams['max_depth'] = int(hyperparams['max_depth'])
    except:
        pass
    print(model, hyperparams)

    model.set_params(**hyperparams)

    cross_val_data = cross_val_score(
        model,
        X_train,
        y_train,
        scoring=['accuracy'],
        cv=3,
        n_jobs=4,
    )
    loss = -cross_val_data.mean()
    print(loss)
    print()
    return loss


# perform the search
trials = Trials()
anneal_search = fmin(
    fn=objective,
    space=param_grid,
    max_evals=50,
    rstate=np.random.RandomState(42),
    algo=anneal.suggest,
    trials=trials,
)
print(anneal_search)
# print(trials.argmin)
print(trials.average_best_error()) # best accuracy 
print(trials.best_trial()) # more info about best model 
