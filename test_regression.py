import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from tree_influence.explainers import LeafInfluence

from tree_influence.explainers import BoostIn


def get_toy_data(dataset, objective, random_state, test_size=0.2):
    data = fetch_california_housing()
    X = data['data']
    y = data['target']
    task = 'regression'
    stratify = y if task in ['binary', 'multiclass'] else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify)
    return X_train, X_test, y_train, y_test


def calc_influence(model, X_train, y_train, X_test, y_test):
    # fit influence estimator
    explainer = BoostIn().fit(model, X_train, y_train)
    inf_values = explainer.get_local_influence(X_test[:10], y_test[:10])
    # print(inf_values.shape)
    # print(inf_values)
    return inf_values

def test_xgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'california_housing', 'regression', 0
    )
    model = XGBRegressor(
        base_score=0.0,  # necessary to reproduce
        reg_lambda=1.0,  # not necessary, but you must not set 0
    )
    model.fit(X_train, y_train)
    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_lgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'california_housing', 'regression', 0
    )
    model = LGBMRegressor()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_cb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'california_housing', 'regression', 0
    )
    model = CatBoostRegressor(
        leaf_estimation_iterations=1
    )
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_shb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'california_housing', 'regression', 0
    )
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


def test_sgb():
    X_train, X_test, y_train, y_test = get_toy_data(
        'california_housing', 'regression', 0
    )
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    inf_values = calc_influence(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    test_sgb()
    test_shb()
    test_xgb()
    test_lgb()
    test_cb()
