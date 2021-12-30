import copy
import logging
import argparse

import tqdm
import numpy as np
import pandas as pd

import sklearn.metrics as skmet
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skprep
import sklearn.model_selection as skmod

from .base import Task, ClassificationScore, load_model

from typing import Any, Dict, Union, Tuple, Callable, Optional

MODEL_MAPPING = {
    'regression': {
        'RandomForest': 'RandomForestRegressor',
        'Linear': 'LinearRegression'
    },
    'classification': {
        'MLP': 'MLPClassifierWithGrad',
        'Linear': 'LogisticRegression'
    }
}


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    cond_nan = np.isinf(y_pred) | np.isnan(y_pred)
    if any(cond_nan):
        y_pred[cond_nan] = 10.0
        logging.warning('Predictions contain NaN or infinity. Setting the metric to 10')

    return skmet.mean_squared_error(y_true, y_pred)


def balanced_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    cond_nan = np.isinf(y_pred) | np.isnan(y_pred)
    if any(cond_nan):
        y_pred[cond_nan] = 0.0
        logging.warning('Predictions contain NaN or infinity. Setting the score to 0')

    return skmet.balanced_accuracy_score(y_true, y_pred)


def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'micro'
) -> float:
    cond_nan = np.isinf(y_pred) | np.isnan(y_pred)
    if any(cond_nan):
        y_pred[cond_nan] = 0.0
        logging.warning('Predictions contain NaN or infinity. Setting the score to 0.')

    return skmet.f1_score(y_true, y_pred, average=average)


def extract_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, np.ndarray]:

    y = df[target_column].values
    X = df.drop(target_column, axis=1)  # noqa

    return X, y


def get_data_splitter(
    num_folds: int,
    random_seed: Optional[int] = None
) -> skmod.ShuffleSplit:

    if num_folds < 1:
        raise ValueError(f'`num_folds` should be positive int, got: {num_folds}')

    split = skmod.ShuffleSplit(n_splits=num_folds, test_size=1.0 / num_folds, random_state=random_seed)

    return split


def make_scorer(
    task: Task = Task.regression,
    classification_score: ClassificationScore = None
) -> Dict[str, Union[str, Callable[[np.ndarray, np.ndarray], float]]]:

    if classification_score is None:
        classification_score = ClassificationScore.balanced_accuracy_score

    greater_is_better = not (task == Task.regression)
    scorer = dict()

    if task == Task.regression:
        scorer['name'] = 'mse'
    else:
        scorer['name'] = classification_score.name

    scorer['func'] = skmet.make_scorer(globals()[scorer['name']], greater_is_better=greater_is_better)

    return scorer


def sanitize_setup(
    args: argparse.Namespace,
    setup: Dict[str, Dict[str, Any]]
):

    def update_params(
        setup: Dict[str, Dict[str, Any]],  # noqa
        model_name: str,  # noqa
        param_key: str,  # noqa
        param_val: Any,  # noqa
        add_param_variable: bool = False
    ) -> Dict[str, Dict[str, Any]]:

        model_params = setup[model_name]

        if 'params_variable' in model_params:
            for param_name in model_params['params_variable'].keys():
                if (param_name == param_key) or add_param_variable:
                    setup[model_name]['params_variable'][param_name] = [param_val]

        if 'params_static' in model_params:
            for param_name in model_params['params_static'].keys():
                if param_name == param_key:
                    setup[model_name]['params_static'][param_name] = param_val

        return setup

    setup_sanitized = copy.deepcopy(setup)

    for model_name in setup_sanitized.keys():
        setup_sanitized = update_params(
            setup=setup_sanitized, model_name=model_name, param_key='random_state', param_val=args.random_seed
        )
        if model_name == 'RandomForest':
            n_est = max(setup_sanitized[model_name]['params_variable']['n_estimators'])

            def check(x):
                return x.shape[0] * (args.num_folds - 1) / args.num_folds >= n_est

            setup_sanitized[model_name]['checker'] = check

    for model_name in list(setup_sanitized.keys()):
        model_name_spec = MODEL_MAPPING[args.task.name].get(model_name, None)
        if model_name_spec is not None:
            setup_sanitized[model_name_spec] = copy.deepcopy(setup_sanitized[model_name])
        del setup_sanitized[model_name]

    return setup_sanitized


def run_cv(
    X: pd.DataFrame,  # noqa
    y: np.ndarray,
    scorer: Dict[str, Union[str, Callable[[np.ndarray, np.ndarray], float]]],
    splitter: skmod.ShuffleSplit,
    setup: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:

    perf_reports = list()
    model_reports = dict()
    feature_scores = dict()

    for model_name, params in tqdm.tqdm(setup.items(), desc='Selecting models', total=len(setup), unit='model'):
        est = load_model(model_name)(**params.get('params_static', dict()))
        if est is None:
            continue

        cv = skpipe.make_pipeline(
            skprep.StandardScaler(),
            skmod.GridSearchCV(
                estimator=est,
                param_grid=params['params_variable'],
                scoring=scorer['func'],
                n_jobs=-1,
                refit=True,
                cv=splitter,
                return_train_score=True
            )
        )
        try:
            cv.fit(X.values, y)
        except Exception as ex:
            logging.exception(ex)
            print(f'Grid search for `{model_name}` failed. Continue')
            continue

        model_reports[model_name] = pd.DataFrame(cv.named_steps.gridsearchcv.cv_results_).sort_values(
            by=f'rank_test_score', axis=0, ascending=True
        ).filter(regex='mean_train|mean_test|std_train|std_test|^params$', axis=1)

        if hasattr(cv.named_steps.gridsearchcv.best_estimator_, 'coef_'):
            feature_scores[model_name] = cv.named_steps.gridsearchcv.best_estimator_.coef_.squeeze()
        if hasattr(cv.named_steps.gridsearchcv.best_estimator_, 'feature_importances_'):
            feature_scores[model_name] = cv.named_steps.gridsearchcv.best_estimator_.feature_importances_

        if feature_scores[model_name] is not None:
            feature_indices_desc = np.argsort(feature_scores[model_name])[::-1]
            feature_names = X.columns[feature_indices_desc].tolist()
            feature_scores[model_name] = feature_scores[model_name][feature_indices_desc]

            df_features = pd.DataFrame({
                'feature': feature_names,
                'score': feature_scores[model_name]
            }).set_index('feature')

            feature_scores[model_name] = df_features.reindex(
                df_features['score'].abs().sort_values(ascending=False).index
            )

        best_model = model_reports[model_name].iloc[0]
        best_model.name = model_name
        perf_reports.append(best_model)

    perf_reports = pd.concat(perf_reports, axis=1, sort=False).T
    perf_reports.sort_values(by='mean_test_score', axis='rows', ascending=False, inplace=True)

    return perf_reports, feature_scores
