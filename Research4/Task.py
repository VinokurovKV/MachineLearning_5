import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):
    def __init__(self, dtype=int):
        self.categories_ = {}
        self.feature_indices_ = {}
        self.dtype = dtype

    def get_params(self, deep=True):
        return {'dtype': self.dtype}

    def fit(self, X):  # Запоминаем уникальные
        self.categories_ = {}
        self.feature_indices_ = {}

        total_columns = 0

        for col in X.columns:
            # Получаем уникальные значения и сортируем их
            unique_vals = sorted(X[col].unique())
            self.categories_[col] = unique_vals
            self.feature_indices_[col] = (total_columns, total_columns + len(unique_vals))
            total_columns += len(unique_vals)

        return self

    def transform(self, X):  # Категориальные -> OHCode

        n_objects = X.shape[0]
        total_columns = sum(len(cats) for cats in self.categories_.values())
        result = np.zeros((n_objects, total_columns), dtype=self.dtype)

        for col in X.columns:
            if col not in self.categories_:
                continue

            start_idx, end_idx = self.feature_indices_[col]
            categories = self.categories_[col]

            for i, category in enumerate(categories):
                mask = X[col] == category
                result[mask, start_idx + i] = 1

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class SimpleCounterEncoder:
    def __init__(self):
        self.category_stats_ = {}

    def fit(self, X, y):  # Запоминаем статистики для категориальных
        self.category_stats_ = {}

        for col in X.columns:
            col_stats = {}
            unique_vals = X[col].unique()

            for val in unique_vals:
                mask = X[col] == val
                y_masked = y[mask]

                if len(y_masked) > 0:
                    successes = y_masked.mean()
                    counters = mask.mean()
                else:
                    successes = 0
                    counters = 0

                col_stats[val] = {
                    'successes': successes,
                    'counters': counters
                }

            self.category_stats_[col] = col_stats

        return self

    def transform(self, X, a=1e-5, b=1e-5):  # Преобразуем категориальные
        n_objects = X.shape[0]
        n_features = len(X.columns)
        result = np.zeros((n_objects, 3 * n_features))

        for col_idx, col in enumerate(X.columns):
            if col not in self.category_stats_:
                continue

            col_stats = self.category_stats_[col]

            for row_idx in range(n_objects):
                val = X.iloc[row_idx][col]

                if val in col_stats:
                    stats = col_stats[val]
                    successes = stats['successes']
                    counters = stats['counters']
                    relation = (successes + a) / (counters + b)
                else:
                    # Если значение не встречалось при обучении
                    successes = 0
                    counters = 0
                    relation = a / b

                result[row_idx, 3 * col_idx] = successes
                result[row_idx, 3 * col_idx + 1] = counters
                result[row_idx, 3 * col_idx + 2] = relation

        return result

    def fit_transform(self, X, y, a=1e-5, b=1e-5):
        return self.fit(X, y).transform(X, a, b)


def group_k_fold(n_objects, n_folds, seed=42):  # Случайно разбиваем на n_folds
    np.random.seed(seed)
    indices = np.arange(n_objects)
    np.random.shuffle(indices)

    fold_size = n_objects // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_objects
        folds.append(indices[start:end])

    return folds


class FoldCounters:
    def __init__(self, n_folds=3):
        self.n_folds = n_folds
        self.folds_ = None
        self.category_stats_ = {}

    def fit(self, X, y, seed=42):  # Запоминаем разбиение и статистики для фолда
        n_objects = X.shape[0]
        self.folds_ = group_k_fold(n_objects, self.n_folds, seed)
        self.category_stats_ = {}

        # Для каждого фолда вычисляем статистики на остальных фолдах
        for fold_idx, test_fold in enumerate(self.folds_):
            train_mask = np.ones(n_objects, dtype=bool)
            train_mask[test_fold] = False

            X_train = X.iloc[train_mask]
            y_train = y.iloc[train_mask] if hasattr(y, 'iloc') else y[train_mask]

            fold_stats = {}

            for col in X.columns:
                col_stats = {}
                unique_vals = X_train[col].unique()

                for val in unique_vals:
                    mask = X_train[col] == val
                    y_masked = y_train[mask]

                    if len(y_masked) > 0:
                        successes = y_masked.mean()
                        counters = mask.mean()
                    else:
                        successes = 0
                        counters = 0

                    col_stats[val] = {
                        'successes': successes,
                        'counters': counters
                    }

                fold_stats[col] = col_stats

            self.category_stats_[fold_idx] = fold_stats

        return self

    def transform(self, X, a=1e-5, b=1e-5):  # Преобразуем с помощью статистики фолда
        n_objects = X.shape[0]
        n_features = len(X.columns)
        result = np.zeros((n_objects, 3 * n_features))

        # Для каждого объекта используем статистики из соответствующего фолда
        for fold_idx, test_fold in enumerate(self.folds_):
            fold_stats = self.category_stats_[fold_idx]

            for obj_idx in test_fold:
                for col_idx, col in enumerate(X.columns):
                    if col not in fold_stats:
                        continue

                    val = X.iloc[obj_idx][col]
                    col_stats = fold_stats[col]

                    if val in col_stats:
                        stats = col_stats[val]
                        successes = stats['successes']
                        counters = stats['counters']
                        relation = (successes + a) / (counters + b)
                    else:
                        # Если значение не встречалось в обучающей части фолда
                        successes = 0
                        counters = 0
                        relation = a / b

                    result[obj_idx, 3 * col_idx] = successes
                    result[obj_idx, 3 * col_idx + 1] = counters
                    result[obj_idx, 3 * col_idx + 2] = relation

        return result

    def fit_transform(self, X, y, seed=42, a=1e-5, b=1e-5):
        return self.fit(X, y, seed).transform(X, a, b)


def weights(x, y):  # Вычисляем оптимальные веса для OHCode при ЛогРегр без свободного члена и сигмоиды
    # Получаем уникальные значения и сортируем их
    unique_vals = sorted(np.unique(x))

    # Для каждого уникального значения вычисляем среднее целевой переменной
    optimal_weights = np.array([y[x == val].mean() if len(y[x == val]) > 0 else 0 for val in unique_vals])

    return optimal_weights
