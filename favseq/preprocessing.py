import logging

import numpy as np
import pandas as pd

import sklearn.linear_model as sklm

from sklearn.impute import KNNImputer
from sklearn.impute._base import _BaseImputer  # noqa
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler

from .base import Task, BalanceStrategy, ImpMethod

from typing import Optional


class LRImputer(_BaseImputer):

    def __init__(self, missing_values=np.nan):
        super().__init__(missing_values=missing_values)
        self.estimators = dict()
        self.scalers = dict()

    def fit(self, X, y=None):  # noqa
        features = X.copy()
        for feat_name in features.columns:

            X, y = features.drop(feat_name, axis=1), features[feat_name]  # noqa

            cond_train = features[feat_name].notna()
            X_train, y_train = X[cond_train], y[cond_train]  # noqa

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)  # noqa

            X_train[np.isnan(X_train)] = 0
            est = sklm.LinearRegression().fit(X_train, y_train.values)

            self.estimators[feat_name] = est
            self.scalers[feat_name] = scaler

        return self

    def transform(self, X):  # noqa
        X_imp = X.copy()  # noqa
        X_imp[:] = np.nan

        for feat_name in X.columns:
            scaler = self.scalers[feat_name]
            est = self.estimators[feat_name]

            y = X[feat_name]
            cond_valid, cond_nan = y.notna(), y.isna()

            X_nan = X.drop(feat_name, axis=1)[cond_nan]  # noqa

            X_nan = scaler.transform(X_nan)  # noqa
            X_nan[np.isnan(X_nan)] = 0
            y_imp = est.predict(X_nan)

            X_imp.loc[cond_valid, feat_name] = y[cond_valid]
            X_imp.loc[cond_nan, feat_name] = y_imp

        return X_imp


def balance_classes(
    df: pd.DataFrame,
    target_column: str,
    strategy: BalanceStrategy
) -> pd.DataFrame:

    if strategy == BalanceStrategy.downsample:
        print('Downsampling the data...')
        genes_dos = df[df[target_column] == 0].index.to_list()
        genes_cmn_dwn = np.random.choice(df[df[target_column] == 1].index.to_list(), size=len(genes_dos))
        df = df.loc[[*genes_cmn_dwn, *genes_dos]]
    elif strategy == BalanceStrategy.upsample:
        print('Oversampling the data...')
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_up, y_up = oversample.fit_resample(df.drop(target_column, axis=1), df[target_column])  # noqa
        df = X_up.join(y_up.to_frame(name=target_column), how='inner')
    else:
        logging.warning('Only one of the class balancing strategies needs to be set. Use: `downsample`, `upsample`')

    return df


def impute_features(
    df: pd.DataFrame,
    imp_method: ImpMethod,
    n_neighbors: Optional[int] = 100
) -> pd.DataFrame:

    print(f'Imputing missing values with `{imp_method.name}`...')
    if imp_method in [ImpMethod.mean, ImpMethod.zero]:
        val_imp: float = df.mean(axis=0) if imp_method == ImpMethod.mean else 0
        df = df.fillna(val_imp, axis=0)
    elif imp_method in [ImpMethod.knn, ImpMethod.lr]:
        features = df.drop(df.filter(regex='^target', axis=1).columns, axis=1)

        if imp_method == ImpMethod.knn:
            features_imp = KNNImputer(n_neighbors=n_neighbors).fit_transform(features.values)
            features_imp = pd.DataFrame.from_records(features_imp, index=features.index, columns=features.columns)
        else:
            features_imp = LRImputer().fit_transform(features)

        df = features_imp.join(df.filter(regex='^target', axis=1), how='inner')

    return df


def prepare_data(
    df: pd.DataFrame,
    task: Task,
    target_column: str,
    fillna: ImpMethod,
    class_balance: BalanceStrategy
) -> pd.DataFrame:

    if task == Task.regression:
        df = df[df.notna().all(axis=1) & df['target_clf'] == 1]

    columns_to_drop = list()
    for column_name in df.columns:
        if column_name.startswith('target') and column_name != target_column:
            columns_to_drop.append(column_name)
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)

    if task == Task.classification:
        if (fillna is not None) and df.isna().values.any():
            df = impute_features(df=df, imp_method=fillna)
        if class_balance is not None:
            df = balance_classes(df=df, target_column=target_column, strategy=class_balance)

    return df
