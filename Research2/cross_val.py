import numpy as np
from collections import defaultdict
import typing


def kfold_split(
    num_objects: int,
    num_folds: int,
    shuffle: bool = False,
    random_state: typing.Optional[int] = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(num_objects)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_size = num_objects // num_folds
    remainder = num_objects % num_folds
    fold_sizes = [fold_size] * num_folds
    if remainder > 0:
        fold_sizes[-1] += remainder

    splits = []
    current = 0
    for size in fold_sizes:
        val_indices = indices[current:current + size]
        train_indices = np.concatenate([indices[:current], indices[current + size:]])
        splits.append((train_indices, val_indices))
        current += size

    return splits


def knn_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, list],
    score_function: callable,
    folds: list[tuple[np.ndarray, np.ndarray]],
    knn_class: object
) -> dict[tuple, float]:
    results = {}
    for normalizer, normalizer_name in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    fold_scores = []
                    for train_idx, val_idx in folds:
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        if normalizer is not None:
                            scaler = normalizer
                            X_train = scaler.fit_transform(X_train)
                            X_val = scaler.transform(X_val)

                        model = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = score_function(y_val, y_pred)
                        fold_scores.append(score)

                    key = (normalizer_name, n_neighbors, metric, weight)
                    results[key] = np.mean(fold_scores)

    return results
