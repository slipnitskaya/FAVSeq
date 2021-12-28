import os
import json
import argparse

import pandas as pd
import matplotlib as mpl

from .base import Task, BalanceStrategy, ImpMethod, ClassificationScore, convert_model_name
from .preprocessing import prepare_data
from .evaluation import extract_target, get_data_splitter, make_scorer, sanitize_setup, run_cv
from .export import plot_performance_summary, store_individual_reports, plot_ranking_results, plot_rfe_scores
from .feature_selection import get_rfe_scores

mpl.rcParams.update({'font.size': 14, 'figure.figsize': (5, 5), 'savefig.bbox': 'tight'})

NUM_FOLDS = 5
RANDOM_SEED = 42


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--path-to-input', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str, required=True)
    parser.add_argument('-t', '--task', type=Task.argtype, choices=Task, default=Task.classification)
    parser.add_argument('-n', '--fillna', type=ImpMethod.argtype, choices=ImpMethod, default=None)
    parser.add_argument('-b', '--class-balance', type=BalanceStrategy.argtype, choices=BalanceStrategy, default=None)
    parser.add_argument(
        '-c', '--classification-score', type=ClassificationScore.argtype, choices=ClassificationScore, default=None
    )
    parser.add_argument('-r', '--random-seed', type=int, default=RANDOM_SEED)
    parser.add_argument('-f', '--num-folds', type=int, default=NUM_FOLDS)
    parser.add_argument('-S', '--save-individual', action='store_true')

    args = parser.parse_args()
    task_id = 'regr' if args.task == Task.regression else 'clf'

    # create an output directory
    out_dir = f'{args.out_dir}/task_{task_id}'
    os.makedirs(out_dir, exist_ok=True)

    # load data
    print(f'Loading `{os.path.basename(args.path_to_input)}`...')
    df = pd.read_csv(args.path_to_input, sep=';', index_col=0)

    df = prepare_data(df, args.task, f'target_{task_id}', args.fillna, args.class_balance)
    setup = json.load(open('protocols.json'))

    # split the data
    X, y = extract_target(df, f'target_{task_id}')  # noqa
    splitter = get_data_splitter(args.num_folds, args.random_seed)

    # define experimental setup
    scorer = make_scorer(args.task, args.classification_score)
    setup = sanitize_setup(args, setup)

    # train and test the models
    reports_summary, reports_individual = run_cv(X, y, scorer, splitter, setup)

    # save performance reports
    reports_summary.to_csv(f'{out_dir}/summary.csv', sep=';')  # noqa
    store_individual_reports(reports_individual, out_dir, args.save_individual)

    # visualize performance summary
    plot_performance_summary(
        summary=reports_summary,
        score_name=scorer["name"],
        out_dir=out_dir
    )

    # rank features using RFE-based-on-best-model
    best_model_name = reports_summary.iloc[0].name
    best_params = {**setup[best_model_name]['params_static'], **reports_summary.iloc[0].params}
    rfe_scores, rfe_ranking = get_rfe_scores(X, y, best_model_name, best_params, scorer, splitter)

    # save RFE summaries
    pd.DataFrame.from_records(
        rfe_scores, index=list(range(1, len(rfe_scores) + 1)), columns=[f'split{n}' for n in range(args.num_folds)]
    ).to_csv(f'{out_dir}/scores_RFE-{convert_model_name(best_model_name)}.csv', sep=';')  # noqa
    rfe_ranking.to_csv(f'{out_dir}/ranking_RFE-{convert_model_name(best_model_name)}.csv', sep=';')  # noqa

    # visualize RFE summary
    plot_rfe_scores(
        rfe_scores=rfe_scores,
        model_name=best_model_name,
        out_dir=out_dir
    )
    plot_ranking_results(
        rfe_ranking,
        model_name=f'RFE-{best_model_name}',
        out_dir=out_dir
    )
    print(f'Process finished successfully. Results are stored in `{out_dir}`.')


if __name__ == '__main__':
    run()
