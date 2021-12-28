import numpy as np
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from .base import load_model

from typing import Dict, Tuple, Union, Callable


def get_rfe_scores(
    X: pd.DataFrame,  # noqa
    y: np.ndarray,
    best_model: str,
    best_params: Dict,
    scorer: Dict[str, Union[str, Callable[[np.ndarray, np.ndarray], float]]],
    splitter: ShuffleSplit
) -> Tuple[np.ndarray, pd.DataFrame]:

    print('Estimating feature importance using RFE...')
    splitter = ShuffleSplit(
        n_splits=splitter.n_splits, test_size=splitter.test_size, random_state=splitter.random_state
    )

    rfe = None
    for train_idx, test_idx in splitter.split(X.values, y):
        X_train, X_test = X.values[train_idx], X.values[test_idx]  # noqa
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  # noqa

        est_best = load_model(best_model)(**best_params)
        rfe = RFECV(estimator=est_best, step=1, cv=splitter, scoring=scorer['func'])

        rfe.fit(X_train, y_train)

    feature_ranks = rfe.ranking_
    feature_indices_asc = np.argsort(feature_ranks)
    feature_names = X.columns[feature_indices_asc].tolist()
    feature_ranks = feature_ranks[feature_indices_asc]

    rfe_ranking = pd.DataFrame({'feature': feature_names, 'score': feature_ranks}).set_index('feature')
    rfe_ranking = rfe_ranking.reindex(rfe_ranking['score'].abs().sort_values(ascending=False).index)
    rfe_scores = -1 * np.asarray(
        [rfe.cv_results_[f'split{i}_test_score'] for i in range(len(rfe.cv_results_) - 2)]
    ).T

    return rfe_scores, rfe_ranking
