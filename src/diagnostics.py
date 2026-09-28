"""Диагностика качества; порог здесь нужен только для разбора ошибок."""

import numpy as np
import pandas as pd

from bot_detection_challenge.metric import precision_at_recall, pr_curve
from src.modeling import SEED, evaluate, predict_configuration


def best_operating_point(y, score):
    precision, recall = pr_curve(y, score)
    allowed = np.flatnonzero(recall >= 0.7)
    best = allowed[np.argmax(precision[allowed])]
    thresholds = np.unique(score)[::-1]
    threshold = float(thresholds[best])
    flagged = np.asarray(score) >= threshold
    y = np.asarray(y)
    return {
        "threshold": threshold,
        "precision": float(precision[best]), "recall": float(recall[best]),
        "tp": int(((y == 1) & flagged).sum()),
        "fp": int(((y == 0) & flagged).sum()),
        "fn": int(((y == 1) & ~flagged).sum()),
        "tn": int(((y == 0) & ~flagged).sum()),
    }


def bootstrap_precision(y, score, repetitions=1000):
    """Куки пересэмплируются вместе с меткой; доля ботов тоже может меняться.

    Это условная оценка разброса на данном отрезке, а не гарантия качества
    на следующей неделе: сдвиг распределения такой bootstrap не моделирует.
    """
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))
        values.append(precision_at_recall(y[indices], score[indices]))
    lower, upper = np.nanquantile(values, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "repetitions": repetitions}


def segment_report(frame, segment, threshold):
    rows = []
    for value, subset in frame.groupby(segment):
        y, score = subset.target.to_numpy(), subset.score.to_numpy()
        flagged = score >= threshold
        tp = int(((y == 1) & flagged).sum())
        fp = int(((y == 0) & flagged).sum())
        row = {
            "segment": str(value), "n": len(subset), "positives": int(y.sum()),
            "tp_fixed_threshold": tp, "fp_fixed_threshold": fp,
            "recall_fixed_threshold": tp / y.sum() if y.sum() else np.nan,
            "precision_fixed_threshold": tp / flagged.sum() if flagged.sum() else np.nan,
        }
        if 0 < y.sum() < len(y):
            row.update(evaluate(y, score))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(y, first, second, repetitions=1000):
    """Разность метрик на одинаковых пересэмплированных куках."""
    y = np.asarray(y)
    first, second = np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))
        differences.append(
            precision_at_recall(y[indices], second[indices])
            - precision_at_recall(y[indices], first[indices])
        )
    return {
        "difference": precision_at_recall(y, second) - precision_at_recall(y, first),
        "lower": float(np.nanquantile(differences, 0.025)),
        "upper": float(np.nanquantile(differences, 0.975)),
    }


def group_permutation_importance(fitted, X, y, groups, repetitions=10):
    """Переставляем группу целиком, сохраняя зависимости внутри неё."""
    original = precision_at_recall(y, predict_configuration(fitted, X))
    used = set()
    for _, columns in fitted:
        used.update(columns)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, group_columns in groups.items():
        columns = [column for column in group_columns if column in used]
        if not columns:
            continue
        differences = []
        for _ in range(repetitions):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            # Присваиваем массив, чтобы pandas не восстановил исходный порядок
            # по индексу cookie_id после перестановки.
            for column in columns:
                shuffled[column] = X[column].to_numpy()[order]
            score = predict_configuration(fitted, shuffled)
            differences.append(original - precision_at_recall(y, score))
        rows.append({
            "group": name, "mean_drop": float(np.mean(differences)),
            "std_drop": float(np.std(differences)), "repetitions": repetitions,
        })
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False)
