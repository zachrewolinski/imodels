import numpy as np
import scipy as sp
import pandas as pd

import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial
from sklearn.metrics import r2_score

from scipy.sparse.linalg import cg
from krylov import cg as krylov_cg

def neg_r2_score(y_true, y_pred):
    return -1 * r2_score(y_true, y_pred)



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



def _trim_values(values, trim=None):
    if trim is not None:
        assert 0 < trim < 0.5, "Limit must be between 0 and 0.5"
        return np.clip(values, trim, 1 - trim)
    else:
        return values



def get_alpha_grid(X, y, start=-5, stop=5, num=100):
    X = X - X.mean(axis=0)
    y = y - y.mean(axis=0)
    sigma_sq_ = np.linalg.norm(y, axis=0) ** 2 / X.shape[0]
    X_var_ = np.linalg.norm(X, axis=0) ** 2
    alpha_opts_ = (X_var_[:, np.newaxis] / (X.T @ y)) ** 2 * sigma_sq_
    base = np.max(alpha_opts_)
    alphas = np.logspace(start, stop, num=num) * base
    return alphas

def fast_hessian_vector_inverse(H,X,tol = 1e-5):
    p, n = X.shape
    inverse_hvps = np.zeros((p, n))

    for i in range(n):
        vector = X[:, i].reshape(-1, 1)
        inverse_hvp, _ = krylov_cg(H, vector, tol=tol)
        inverse_hvps[:, i] = inverse_hvp.flatten()

    return inverse_hvps


def count_sketch_inverse(A, B, num_sketches=10):
    """
    Compute the approximate product of A^(-1) and B using CountSketch.

    Parameters:
    A (numpy.ndarray): The square matrix to be inverted.
    B (numpy.ndarray): The matrix to be multiplied with A^(-1).
    num_sketches (int): The number of sketches to use.

    Returns:
    numpy.ndarray: The approximate product A^(-1) B.
    """
    n, _ = A.shape
    _, p = B.shape

    # Initialize the sketch matrices
    S = np.zeros((num_sketches, n, p))

    for k in range(num_sketches):
        # Generate random hash functions for rows and columns
        row_hash = np.random.choice([-1, 1], size=n)
        col_hash = np.random.choice([-1, 1], size=n)

        # Compute the sketch matrix for the current iteration
        sketch_A = np.zeros((n, n))
        sketch_B = np.zeros((n, p))
        for i in range(n):
            for j in range(n):
                sketch_A[i, j] = row_hash[i] * A[i, j] * col_hash[j]
            for j in range(p):
                sketch_B[i, j] = row_hash[i] * B[i, j]

        # Compute the inverse of the sketch of A
        sketch_inv = np.linalg.inv(sketch_A)

        # Compute the approximate matrix product
        S[k] = np.dot(sketch_inv, sketch_B)

    # Compute the median of the sketches
    AB_approx = np.median(S, axis=0)

    return AB_approx
