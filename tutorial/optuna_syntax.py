# optuna syntax 
import optuna
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.ensemble import RandomForestClassifier 

# define the objective function 
def objective(trial):
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 100, 1000)
    rf_max_depth = trial.suggest_int('rf_max_depth', 1, 4)

    model = RandomForestClassifier(
        max_depth = rf_max_depth, n_estimators = rf_n_estimators
    )
    score = cross_val_score(model, 
    X_train,
    y_train, 
    cv = 3)
    accuracy = score.mean()
    return accuracy

study = optuna.create_study(direction = 'maximize',
    sampler = optuna.samplers.RandomSampler(),
    # can change sampler
    # e.g., GridSampler(search_space) passing in a dict 
    # default optuna.samplers.TPESampler() 
 ) 
# create_study(
#    storage: url to a database, sampler: default tpe, pruner: default medianpruner 
#)
study.optimize(objective, n_trials = 10)
# suggest_categorical, suggest_discrete_uniform, etc. 