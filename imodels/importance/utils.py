import copy
import numpy as np
import pandas as pd
import shap
import lime
import pprint
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import reduce


def _fast_r2_score(y_true, y_pred, multiclass=False):
    """
    Evaluates the r-squared value between the observed and estimated responses.
    Equivalent to sklearn.metrics.r2_score but without the robust error
    checking, thus leading to a much faster implementation (at the cost of
    this error checking). For multi-class responses, returns the mean
    r-squared value across each column in the response matrix.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted responses.
    multiclass: bool
        Whether or not the responses are multi-class.

    Returns
    -------
    Scalar quantity, measuring the r-squared value.
    """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    if multiclass:
        return np.mean(1 - numerator / denominator)
    else:
        return 1 - numerator / denominator


def _neg_log_loss(y_true, y_pred):
    """
    Evaluates the negative log-loss between the observed and
    predicted responses.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted probabilies.

    Returns
    -------
    Scalar quantity, measuring the negative log-loss value.
    """
    return -log_loss(y_true, y_pred)