import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from .base import Task, load_model

from typing import Dict, Tuple, Union, Callable


def get_rfe_scores(
    X: pd.DataFrame,  # noqa
    y: np.ndarray,
    task: Task,
    best_model: str,
    best_params: Dict,
    scorer: Dict[str, Union[str, Callable[[np.ndarray, np.ndarray], float]]],
    splitter: ShuffleSplit
) -> Tuple[np.ndarray, pd.DataFrame]:

    print('Estimating feature importance using RFE...')
    splitter = ShuffleSplit(
        n_splits=splitter.n_splits, test_size=splitter.test_size, random_state=splitter.random_state
    )
    n_features = X.shape[-1]

    feature_ranks = np.ones((splitter.n_splits, n_features, n_features))
    feature_scores = np.zeros((splitter.n_splits, n_features))
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X.values, y)):
        X_train, X_test = X.values[train_idx], X.values[test_idx]  # noqa
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  # noqa
        X_test = scaler.transform(X_test)  # noqa

        for n_features_to_select in np.arange(n_features) + 1:
            est_best = load_model(best_model)(**best_params)
            rfe = RFE(estimator=est_best, n_features_to_select=n_features_to_select, step=1)

            rfe.fit(X_train, y_train)

            feature_ranks[split_idx, n_features_to_select - 1] = rfe.ranking_
            feature_scores[split_idx, n_features_to_select - 1] = scorer['func'](rfe, X_test, y_test)

    feature_ranks = feature_ranks.mean(axis=(0, 1))
    feature_indices_asc = np.argsort(feature_ranks)
    feature_names = X.columns[feature_indices_asc].tolist()
    feature_ranks = feature_ranks[feature_indices_asc]

    rfe_ranking = pd.DataFrame({'feature': feature_names, 'score': feature_ranks ** -1.0}).set_index('feature')
    rfe_ranking = rfe_ranking.reindex(rfe_ranking['score'].abs().sort_values(ascending=False).index)
    rfe_scores = feature_scores.T

    if task == Task.regression:
        rfe_scores *= -1

    return rfe_scores, rfe_ranking
