from typing import Any, Dict, List, Optional, Tuple, Union
from .utils import (
    ClassifierType,
    write_pickle,
)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_esc(
    df_train: pd.DataFrame,
    top_level_label_col_name: str = "top_level_label",
    granular_label_col_name: str = "custom_label",
    features_col_name: str = "classical_features",
) -> Dict[str, Any]:
    def prepare_data(df: pd.DataFrame, label: str, granular_label_col_name: str = "custom_label", features_col_name: str = "classical_features") -> Tuple[np.array, np.array]:
        HUMAN_LABEL = "human"
        if label == HUMAN_LABEL:
            df_use = df
        else:
            df_label = df[df[granular_label_col_name] == label]
            df_human = df[df[granular_label_col_name] == HUMAN_LABEL]
            df_human_sampled = df_human.sample(n=df_label.shape[0])
            df_use = pd.concat([df_label, df_human_sampled])

        X = pd.DataFrame(df_use[features_col_name].tolist()).select_dtypes(exclude=["object"]).fillna(value=0.0).to_numpy()
        y = df_use[granular_label_col_name].apply(lambda x: 1 if x == label else 0).to_numpy()

        return X, y

    all_labels = df_train[granular_label_col_name].unique().tolist()
    clf_lst = []
    for label in all_labels:
        X, y = prepare_data(df_train, label, granular_label_col_name, features_col_name)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X, y)
        clf_lst.append(clf)
    
    clfs = dict(zip(all_labels, clf_lst))
    map_of_granular_label_to_top_level_label = dict(zip(df_train[granular_label_col_name], df_train[top_level_label_col_name]))
    
    model = {
        "clfs": clfs,
        "map_of_granular_label_to_top_level_label": map_of_granular_label_to_top_level_label,
    }
    
    return model


def predict_esc(
    model: Dict[str, RandomForestClassifier],
    df_test: pd.DataFrame,
    features_col_name: str = "classical_features",
) -> pd.DataFrame:
    X = pd.DataFrame(df_test[features_col_name].tolist()).select_dtypes(exclude=["object"]).fillna(value=0.0).to_numpy()

    preds_dct = {}
    for label, clf in model["clfs"].items():
        preds = clf.predict_proba(X)[:, 1]
        preds_dct[label] = preds
    
    preds_df = pd.DataFrame.from_dict(preds_dct, orient="columns")
    preds_df["granular_label_pred_y"] = preds_df.idxmax(axis=1)
    preds_df["granular_label_pred_proba_y"] = preds_df.max(axis=1)
    map_of_granular_label_to_top_level_label = model["map_of_granular_label_to_top_level_label"]
    preds_df["top_level_label_pred_y"] = preds_df["granular_label_pred_y"].apply(lambda x: map_of_granular_label_to_top_level_label[x])
    preds_df["top_level_label_pred_proba_y"] = preds_df["granular_label_pred_proba_y"]

    return preds_df


def score_esc(
    df_test: pd.DataFrame,
    preds_df: pd.DataFrame,
    top_level_label_col_name: str = "top_level_label",
    granular_label_col_name: str = "custom_label",
    granular_label_pred_y_col_name: str = "granular_label_pred_y",
    granular_label_pred_proba_y_col_name: str = "granular_label_pred_proba_y",
    top_level_label_pred_y_col_name: str = "top_level_label_pred_y",
    top_level_label_pred_proba_y_col_name: str = "top_level_label_pred_proba_y",
) -> Dict[str, Any]:
    HUMAN_LABEL = "human"
    BOT_LABEL = "bot"

    top_level_label_true_y = df_test[top_level_label_col_name].tolist()
    top_level_label_true_proba_y = [1 if a == BOT_LABEL else 0 for a in top_level_label_true_y]
    granular_label_true_y = df_test[granular_label_col_name].tolist()
    
    granular_label_pred_y = preds_df[granular_label_pred_y_col_name].tolist()
    granular_label_pred_proba_y = preds_df[granular_label_pred_proba_y_col_name].tolist()
    top_level_label_pred_y = preds_df[top_level_label_pred_y_col_name].tolist()
    top_level_label_pred_proba_y = preds_df[top_level_label_pred_proba_y_col_name].tolist()
    top_level_label_pred_proba_y = [b if a == BOT_LABEL else 1-b for a, b in zip(top_level_label_true_y, top_level_label_pred_proba_y)]

    top_level_scores = {
        "accuracy": accuracy_score(top_level_label_true_y, top_level_label_pred_y),
        "f1": f1_score(top_level_label_true_y, top_level_label_pred_y, average=None),
        "f1_micro": f1_score(top_level_label_true_y, top_level_label_pred_y, average="micro"),
        "f1_macro": f1_score(top_level_label_true_y, top_level_label_pred_y, average="macro"),
        "precision": precision_score(top_level_label_true_y, top_level_label_pred_y, pos_label=BOT_LABEL),
        "recall": recall_score(top_level_label_true_y, top_level_label_pred_y, pos_label=BOT_LABEL),
        "roc_auc": roc_auc_score(top_level_label_true_proba_y, top_level_label_pred_proba_y),
    }

    granular_scores = {
        "accuracy": accuracy_score(granular_label_true_y, granular_label_pred_y),
        "f1": f1_score(granular_label_true_y, granular_label_pred_y, average=None),
        "f1_micro": f1_score(granular_label_true_y, granular_label_pred_y, average="micro"),
        "f1_macro": f1_score(granular_label_true_y, granular_label_pred_y, average="macro"),
        "precision": precision_score(granular_label_true_y, granular_label_pred_y, average=None),
        "precision_micro": precision_score(granular_label_true_y, granular_label_pred_y, average="micro"),
        "precision_macro": precision_score(granular_label_true_y, granular_label_pred_y, average="macro"),
        "recall": recall_score(granular_label_true_y, granular_label_pred_y, average=None),
        "recall_micro": recall_score(granular_label_true_y, granular_label_pred_y, average="micro"),
        "recall_macro": recall_score(granular_label_true_y, granular_label_pred_y, average="macro"),
        # "roc_auc": roc_auc_score(granular_label_true_y, granular_label_pred_proba_y),
    }

    scores = {
        "top_level_scores": top_level_scores,
        "granular_scores": granular_scores,
    }

    return scores


def make_train_test_experiment_object(
    clf: Union[ClassifierType, GridSearchCV, RandomizedSearchCV],
    train_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
    scores: Dict[str, Any],
) -> Dict[str, Any]:
    experiment = {
        "dataset_metadata": {
            "train_dataset": train_dataset["dataset_metadata"],
            "test_dataset": test_dataset["dataset_metadata"],
        },
        "clf": clf,
        "scores": scores,
    }

    return experiment


def train_test_experiment(
    clf: ClassifierType,
    train_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
    is_write_to_disk: bool = False,
    path: str = "train_test_experiment.pkl",
) -> Dict[str, Any]:
    clf = train_classifier(clf, train_dataset)
    scores = score_classifier(clf, test_dataset)

    experiment = make_train_test_experiment_object(clf, train_dataset, test_dataset, scores)

    if is_write_to_disk:
        write_pickle(experiment, path)

    return experiment


def score_classifier(clf: Union[ClassifierType, GridSearchCV, RandomizedSearchCV], dataset: Dict[str, Any]) -> Dict[str, Any]:
    test_dataset = dataset
    test_X, test_y = test_dataset["X"], test_dataset["y"]
    pred_y = clf.predict(test_X)
    pred_proba_y = clf.predict_proba(test_X)[:, 1]

    scores = {
        "accuracy": accuracy_score(test_y, pred_y),
        "f1": f1_score(test_y, pred_y, average="binary"),
        "f1_micro": f1_score(test_y, pred_y, average="micro"),
        "f1_macro": f1_score(test_y, pred_y, average="macro"),
        "precision": precision_score(test_y, pred_y),
        "recall": recall_score(test_y, pred_y),
        "roc_auc": roc_auc_score(test_y, pred_proba_y),
    }

    return scores


def train_classifier(clf: ClassifierType, dataset: Dict[str, Any]) -> ClassifierType:
    train_dataset = dataset
    train_X, train_y = train_dataset["X"], train_dataset["y"]

    clf.fit(train_X, train_y)

    return clf


def train_classifier_with_model_selection(clf: ClassifierType, dataset: Dict[str, Any], param_grid: Dict[str, List[Any]], n_fits: Optional[int] = None) -> Union[GridSearchCV, RandomizedSearchCV]:
    train_dataset = dataset
    train_X, train_y = train_dataset["X"], train_dataset["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    if n_fits is None:
        model_selection = GridSearchCV(estimator=clf, param_grid=param_grid, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=1, refit=True)
    else:
        model_selection = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=n_fits, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=1, refit=True)
    model_selection.fit(train_X, train_y)

    return model_selection


def train_cv_experiment(clf: ClassifierType, dataset: Dict[str, Any], is_write_to_disk: bool = False, path: str = "train_cv_experiment.pkl", verbose: int = 0) -> Dict[str, Any]:
    X, y = dataset["X"], dataset["y"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=None)
    scores = cross_validate(clf, X, y, scoring=["accuracy", "f1", "f1_micro", "f1_macro", "precision", "recall", "roc_auc"], cv=cv, n_jobs=-1, verbose=verbose)

    training_run = {
        "dataset_metadata": dataset["dataset_metadata"],
        "clf": clf,
        "scores": scores,
    }

    if is_write_to_disk:
        write_pickle(training_run, path)

    return training_run
