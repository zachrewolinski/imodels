import numpy as np
import scipy as sp
import pandas as pd

import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial


def _extract_coef_and_intercept(estimator):
    """
    Get the coefficient vector and intercept from a GLM estimator
    """
    coef_ = estimator.coef_
    intercept_ = estimator.intercept_
    if coef_.ndim > 1:  # For classifer estimators
        coef_ = coef_.ravel()
        intercept_ = intercept_[0]
    augmented_coef_ = np.append(coef_, intercept_)
    return augmented_coef_


def _set_alpha(estimator, alpha):
    if hasattr(estimator, "alpha"):
        estimator.set_params(alpha=alpha)
    elif hasattr(estimator, "C"):
        estimator.set_params(C=1/alpha)
    else:
        warnings.warn("Estimator has no regularization parameter.")


def _get_preds(data_block, coefs, inv_link_fn, intercept=None):
    if coefs.ndim > 1: # LOO predictions
        if coefs.shape[1] == (data_block.shape[1] + 1):
            intercept = coefs[:, -1]
            coefs = coefs[:, :-1]
        lin_preds = np.sum(data_block * coefs, axis=1) + intercept
    else:
        if len(coefs) == (data_block.shape[1] + 1):
            intercept = coefs[-1]
            coefs = coefs[:-1]
        lin_preds = data_block @ coefs + intercept
    return inv_link_fn(lin_preds)


def _trim_values(values, trim=None):
    if trim is not None:
        assert 0 < trim < 0.5, "Limit must be between 0 and 0.5"
        return np.clip(values, trim, 1 - trim)
    else:
        return values



def huber_loss(y, preds, epsilon=1.35):
    """
    Evaluates Huber loss function.

    Parameters
    ----------
    y: array-like of shape (n,)
        Vector of observed responses.
    preds: array-like of shape (n,)
        Vector of estimated/predicted responses.
    epsilon: float
        Threshold, determining transition between squared
        and absolute loss in Huber loss function.

    Returns
    -------
    Scalar value, quantifying the Huber loss. Lower loss
    indicates better fit.

    """
    total_loss = 0
    for i in range(len(y)):
        sample_absolute_error = np.abs(y[i] - preds[i])
        if sample_absolute_error < epsilon:
            total_loss += 0.5 * ((y[i] - preds[i]) ** 2)
        else:
            sample_robust_loss = epsilon * sample_absolute_error - 0.5 * \
                                 epsilon ** 2
            total_loss += sample_robust_loss
    return total_loss / len(y)


def get_alpha_grid(X, y, start=-5, stop=5, num=100):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas