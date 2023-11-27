"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.14
"""
from datetime import datetime
import logging
from typing import Dict

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series):
    lr = LinearRegression()
    rf = RandomForestRegressor()
    svr = SVR()
    xgbr = XGBRegressor()
    lgbmr = LGBMRegressor()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svr.fit(X_train, y_train)
    xgbr.fit(X_train, y_train)
    lgbmr.fit(X_train, y_train)

    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    experiment_name = f"experiment_{now}"
    mlflow.create_experiment(name=experiment_name)
    experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))['experiment_id']
    mlflow.autolog()

    with mlflow.start_run(run_name="LR", experiment_id=experiment_id):
        y_pred = lr.predict(X_valid)
        mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, y_pred))
        mlflow.log_params(lr.get_params())
        mlflow.sklearn.log_model(lr, "model")

    with mlflow.start_run(run_name="RF", experiment_id=experiment_id):
        y_pred = rf.predict(X_valid)
        mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, y_pred))
        mlflow.log_params(rf.get_params())
        mlflow.sklearn.log_model(rf, "model")


    with mlflow.start_run(run_name="SVR", experiment_id=experiment_id):
        y_pred = svr.predict(X_valid)
        mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, y_pred))
        mlflow.log_params(svr.get_params())
        mlflow.sklearn.log_model(svr, "model")

    with mlflow.start_run(run_name="XGB", experiment_id=experiment_id):
        y_pred = xgbr.predict(X_valid)
        mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, y_pred))
        mlflow.log_params(xgbr.get_params())
        mlflow.sklearn.log_model(xgbr, "model")

    with mlflow.start_run(run_name="LGBM", experiment_id=experiment_id):
        y_pred = lgbmr.predict(X_valid)
        mlflow.log_metric("valid_mae", mean_absolute_error(y_valid, y_pred))
        mlflow.log_params(lgbmr.get_params())
        mlflow.sklearn.log_model(lgbmr, "model")

    return get_best_model(experiment_id)



def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
