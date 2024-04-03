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

def _get_loo_coefficients(X, y, orig_coef_,alpha,inv_link_fn,l_doubledot,r_doubledot, l_dot,max_h=1-1e-4):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """

        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X.shape[0]
        orig_preds = _get_preds(X, orig_coef_, inv_link_fn)
        support_idxs = orig_coef_ != 0
        if not any(support_idxs):
            return orig_coef_ * np.ones_like(X1)
        X1 = X1[:, support_idxs]
        orig_coef_ = orig_coef_[support_idxs]
        l_doubledot_vals = l_doubledot(y, orig_preds)/n
        J = X1.T * l_doubledot_vals @ X1
        if r_doubledot is not None:
            r_doubledot_vals = r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0 # Do not penalize constant term
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        
        normal_eqn_mat = np.linalg.inv(J) @ X1.T
        
        h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals/n
        h_vals[h_vals == 1] = max_h
        l_dot_vals = l_dot(y, orig_preds)/n
        influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)

        loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix

        if not all(support_idxs):
            loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            loo_coef_dense_[support_idxs, :] = loo_coef_
            loo_coef_ = loo_coef_dense_
        return loo_coef_.T,influence_matrix