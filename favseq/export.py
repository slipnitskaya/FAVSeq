import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import ClassificationScore, convert_model_name

from typing import Dict, Optional


def plot_performance_summary(
    summary: pd.DataFrame,
    score_name: str,
    out_dir: str
) -> None:

    df_std = summary[['std_test_score', 'std_train_score']].values.T
    summary = summary[['mean_test_score', 'mean_train_score']]
    summary = summary.rename(columns={'mean_test_score': 'test data', 'mean_train_score': 'train data'})
    summary.index = [convert_model_name(mn) for mn in summary.index]

    if score_name in map(str, ClassificationScore):
        y_label = ' '.join([str(w).capitalize() for w in score_name.split('_')])
        legend_loc = 'lower center'
        y_lim = {'bottom': summary.values.reshape(-1).min() * 0.95, 'top': summary.values.reshape(-1).max() * 1.05}
    else:
        summary = summary * -1
        y_label = score_name.upper()
        legend_loc = 'upper right'
        y_lim = {'bottom': 0.0}

    ax = summary.plot(kind='bar', yerr=df_std, style=None, rot=0, xlabel='models', ylabel=y_label)  # noqa
    ax.set_ylim(**y_lim)

    ax.legend(loc=legend_loc)
    ax.grid(which='major', axis='y', alpha=0.3)

    plt.savefig(os.path.join(out_dir, 'performance'))
    plt.close()


def store_individual_reports(
    results_individual: Dict[str, pd.DataFrame],
    out_dir: str,
    save: bool = False
) -> None:

    for model_name, feature_scores in results_individual.items():
        if save:
            path_to_rank = os.path.join(out_dir, f'ranking_{convert_model_name(model_name)}.csv')
            feature_scores.to_csv(path_to_rank)  # noqa

            plot_ranking_results(
                feature_scores,
                model_name=model_name,
                out_dir=out_dir
            )


def plot_ranking_results(
    feature_scores: pd.DataFrame,
    model_name: str,
    out_dir: str,
    n_top: Optional[int] = None
) -> None:

    n_top = 10 if n_top is None else n_top
    feature_scores = feature_scores.head(n_top)
    feature_scores = feature_scores.sort_values(by='score', ascending=True)

    feature_scores.plot(kind='barh', legend=False, rot=0)
    plt.xlabel('relevance')
    plt.ylabel(None)
    plt.savefig(os.path.join(out_dir, f'ranking_{convert_model_name(model_name)}'))
    plt.close()


def plot_rfe_scores(
    rfe_scores: np.ndarray,
    model_name: str,
    out_dir: str
) -> None:

    n_subset_feats = list(range(1, len(rfe_scores) + 1))
    rfe_scores_mean, rfe_scores_std = rfe_scores.mean(axis=1), rfe_scores.std(axis=1)

    plt.figure()
    plt.plot(n_subset_feats, rfe_scores_mean)
    plt.fill_between(
        n_subset_feats, rfe_scores_mean - rfe_scores_std, rfe_scores_mean + rfe_scores_std, color='C0', alpha=0.2
    )
    plt.xlabel('feature subset size')
    plt.ylabel(f'CV objective')

    plt.savefig(os.path.join(out_dir, f'scores_{convert_model_name(model_name)}'))
    plt.close()
