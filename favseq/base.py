import logging
import argparse

import sklearn.ensemble
import sklearn.linear_model

import favseq.mlp as mlp

from enum import auto, Enum


EST_PACKAGES = [
    mlp,
    sklearn.ensemble,
    sklearn.linear_model
]

MODEL_MAPPING_INVERSE = {
    'LinearRegression': 'LR',
    'LogisticRegression': 'LR',
    'RandomForestRegressor': 'RF',
    'MLPClassifierWithGrad': 'MLP'
}


class ArgTypeMixin(Enum):

    @classmethod
    def argtype(cls, s: str) -> Enum:
        try:
            return cls[s]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{s!r} is not a valid {cls.__name__}"
            )

    def __str__(self):
        return self.name


class Task(ArgTypeMixin, Enum):
    regression = auto()
    classification = auto()


class BalanceStrategy(ArgTypeMixin, Enum):
    upsample = auto()
    downsample = auto()


class ImpMethod(ArgTypeMixin, Enum):
    lr = auto()
    knn = auto()
    mean = auto()
    zero = auto()


class ClassificationScore(ArgTypeMixin, Enum):
    balanced_accuracy_score = auto()
    f1_score = auto()


def convert_model_name(model_name: str) -> str:
    *prefix, model_name = model_name.split('-', 1)
    model_name_short = '-'.join(prefix + [MODEL_MAPPING_INVERSE[model_name]])

    return model_name_short


def load_model(model_name: str):
    est = None
    for pack in EST_PACKAGES:
        try:
            est = getattr(pack, model_name)
        except AttributeError:
            pass

    if est is None:
        try:
            return globals()[model_name]
        except KeyError:
            logging.warning(f'Model `{model_name}` was not found')

    return est
