"""Создание моделей, обучение, прогнозы и проверка формата submission."""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from bot_detection_challenge.metric import precision_at_recall, recall_at_fpr


SEED = 42
THREADS = 6
CAT_DEFAULTS = {
    "iterations": 700, "depth": 5, "learning_rate": 0.05,
    "l2_leaf_reg": 5, "loss_function": "Logloss", "random_seed": SEED,
    "thread_count": THREADS, "task_type": "CPU", "verbose": False,
    "allow_writing_files": False,
}


def feature_columns(config, groups):
    if "members" in config:
        columns = []
        for member in config["members"]:
            for column in feature_columns(member, groups):
                if column not in columns:
                    columns.append(column)
        return columns
    columns = []
    for group in config["groups"]:
        columns.extend(groups[group])
    excluded = config.get("exclude_columns", [])
    return [column for column in columns if column not in excluded]


def make_model(config):
    if config["model"] == "random_forest":
        return RandomForestClassifier(**({
            "n_estimators": 300, "min_samples_leaf": 3,
            "random_state": 0, "n_jobs": THREADS,
        } | config.get("params", {})))
    if config["model"] == "catboost":
        return CatBoostClassifier(**(CAT_DEFAULTS | config.get("params", {})))
    if config["model"] == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**({
            "max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "l2_regularization": 5, "early_stopping": False, "random_state": SEED,
        } | config.get("params", {})))
    raise ValueError(f"Неизвестная модель: {config['model']}")


def fit_model(config, X, y):
    model = make_model(config)
    categorical = list(X.select_dtypes(exclude="number").columns)
    if config["model"] == "catboost":
        model.fit(X, y, cat_features=categorical)
    elif config["model"] == "hist_gradient_boosting":
        # Словарь категорий обучается только на тренировочной части фолда.
        preprocessing = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ], remainder="passthrough")
        model = Pipeline([("preprocessing", preprocessing), ("model", model)])
        with threadpool_limits(limits=THREADS):
            model.fit(X, y)
    else:
        model.fit(X, y)
    return model


def fit_configuration(config, X, y, groups):
    """Одна модель или простое усреднение нескольких явно заданных моделей."""
    fitted = []
    for member in config.get("members", [config]):
        columns = feature_columns(member, groups)
        model = fit_model(member, X[columns], y)
        fitted.append((model, columns))
    return fitted


def predict_configuration(fitted, X):
    predictions = []
    with threadpool_limits(limits=THREADS):
        for model, columns in fitted:
            predictions.append(model.predict_proba(X[columns])[:, 1])
    return np.mean(predictions, axis=0)


def evaluate(y, score):
    return {
        "precision_at_recall70": precision_at_recall(y, score),
        "average_precision": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "recall_at_fpr01": recall_at_fpr(y, score),
    }


def validate_submission(submission, test):
    if list(submission.columns) != ["cookie_id", "score"]:
        raise ValueError("Нужны ровно две колонки: cookie_id, score")
    if not submission.cookie_id.is_unique or len(submission) != len(test):
        raise ValueError("Некорректное число куки или дубликаты")
    if not submission.cookie_id.reset_index(drop=True).equals(test.cookie_id.reset_index(drop=True)):
        raise ValueError("Куки или порядок строк не совпадают с test.csv")
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError("Оценки должны быть конечными числами от 0 до 1")
