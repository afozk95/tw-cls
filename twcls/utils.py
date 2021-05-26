from typing import Any, Dict, List, Union
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle


ClassifierType = Union[
    RandomForestClassifier,
    GradientBoostingClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
]


def get_classifier(classifier_name: str, **kwargs: Any) -> ClassifierType:
    if classifier_name == "sklearn.random_forest_classifier":
        return get_sklearn_random_forest_classifier(**kwargs)
    elif classifier_name == "sklearn.gradient_boosting_classifier":
        return get_sklearn_gradient_boosting_classifier(**kwargs)
    elif classifier_name == "lgbm":
        return get_lgbm_classifier(**kwargs)
    elif classifier_name == "xgb":
        return get_xgb_classifier(**kwargs)
    elif classifier_name == "catboost":
        return get_catboost_classifier(**kwargs)
    else:
        raise ValueError(f"Unknown classifier_name = {classifier_name}")


def get_classifier_param_grid(classifier_name: str) -> Dict[str, List[Any]]:
    if classifier_name == "sklearn.random_forest_classifier":
        return get_sklearn_random_forest_classifier_param_grid()
    elif classifier_name == "sklearn.gradient_boosting_classifier":
        return get_sklearn_gradient_boosting_classifier_param_grid()
    elif classifier_name == "lgbm":
        return get_lgbm_classifier_param_grid()
    elif classifier_name == "xgb":
        return get_xgb_classifier_param_grid()
    elif classifier_name == "catboost":
        return get_catboost_classifier_param_grid()
    else:
        raise ValueError(f"Unknown classifier_name = {classifier_name}")


def get_sklearn_random_forest_classifier(**kwargs: Any) -> RandomForestClassifier:
    clf = RandomForestClassifier(**kwargs)
    return clf


def get_sklearn_gradient_boosting_classifier(**kwargs: Any) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(**kwargs)
    return clf


def get_lgbm_classifier(**kwargs: Any) -> LGBMClassifier:
    clf = LGBMClassifier(**kwargs)
    return clf


def get_xgb_classifier(**kwargs: Any) -> XGBClassifier:
    clf = XGBClassifier(**kwargs, use_label_encoder=False)
    return clf


def get_catboost_classifier(**kwargs: Any) -> CatBoostClassifier:
    clf = CatBoostClassifier(**kwargs)
    return clf


def get_sklearn_random_forest_classifier_param_grid() -> Dict[str, List[Any]]:
    param_grid = {
        "bootstrap": [True, False],
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "max_features": ["auto", "sqrt"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [20, 50, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    }
    return param_grid


def get_sklearn_gradient_boosting_classifier_param_grid() -> Dict[str, List[Any]]:
    param_grid = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6],
    }
    return param_grid


def get_lgbm_classifier_param_grid() -> Dict[str, List[Any]]:
    param_grid = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6],
    }
    return param_grid


def get_xgb_classifier_param_grid() -> Dict[str, List[Any]]:
    param_grid = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6],
    }
    return param_grid


def get_catboost_classifier_param_grid() -> Dict[str, List[Any]]:
    param_grid = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6],
    }
    return param_grid


def write_pickle(data: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data
