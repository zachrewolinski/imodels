import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import log_loss, mean_squared_error
from scipy.special import softmax
from imodels.tree.rf_plus.ppms.ppms_util import _extract_coef_and_intercept, _set_alpha, _get_preds, _trim_values, huber_loss

class PartialPredictionModelBase(ABC):
    """
    An interface for partial prediction models, objects that make use of a
    block partitioned data object, fits a regression or classification model
    on all the data, and for each block k, applies the model on a modified copy
    of the data (by either imputing the mean of each feature in block k or
    imputing the mean of each feature not in block k.)

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions.
    """

    def __init__(self, estimator):
        # print("PartialPredictionModelBase init")
        self.estimator = copy.deepcopy(estimator)
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the partial prediction model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        """
        self._fit_model(X, y)
        self.is_fitted = True

    @abstractmethod
    def _fit_model(self, X, y):
        """
        Fit the regression or classification model on all the data.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data using the fitted model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.
        """
        pass

    @abstractmethod
    def predict_full(self, blocked_data):
        """
        Make predictions using all the data based upon the fitted model.
        Used to make full predictions in MDI+.

        Parameters
        ----------
        blocked_data: BlockPartitionedData object
            The block partitioned covariate data, for which to make predictions.
        """
        pass

    @abstractmethod
    def predict_partial_k(self, blocked_data, k, mode):
        """
        Make predictions on modified copies of the data based on the fitted model,
        for a particular feature k of interest. Used to get partial predictions
        for feature k in MDI+.

        Parameters
        ----------
        blocked_data: BlockPartitionedData object
            The block partitioned covariate data, for which to make predictions.
        k: int
            Index of feature in X of interest.
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k, "keep_rest" imputes the mean of each feature in block k
        """
        pass

    def predict_partial(self, blocked_data, mode, zero_values=None):
        """
        Make predictions on modified copies of the data based on the fitted model,
        for each feature under study. Used to get partial predictions in MDI+.

        Parameters
        ----------
        blocked_data: BlockPartitionedData object
            The block partitioned covariate data, for which to make predictions.
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k, "keep_rest" imputes the mean of each feature in block k
        zero_values: ndarray of shape (n_features, ) representing the value of
            each column that should be treated as a zero value. If None, then
            we do not use these.

        Returns
        -------
        List of length n_features of partial predictions for each feature.
        """
        # print("IN 'predict_partial' method of PartialPredictionModelBase")
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode)
        return partial_preds


class _GenericPPM(PartialPredictionModelBase, ABC):
    """
    Partial prediction model for arbitrary estimators. May be slow.
    """

    def __init__(self, estimator):
        super().__init__(estimator)

    def _fit_model(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_full(self, blocked_data):
        return self.predict(blocked_data.get_all_data())

    def predict_partial_k(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.predict(modified_data)


class GenericRegressorPPM(_GenericPPM, PartialPredictionModelBase, ABC):
    """
    Partial prediction model for arbitrary regression estimators. May be slow.
    """
    ...


class GenericClassifierPPM(_GenericPPM, PartialPredictionModelBase, ABC):
    """
    Partial prediction model for arbitrary classification estimators. May be slow.
    """

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict_partial_k(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.predict_proba(modified_data)


class _GlmPPM(PartialPredictionModelBase, ABC):
    """
    PPM class for GLM estimator. The GLM estimator is assumed to have a single
    regularization hyperparameter accessible as a named attribute called either
    "alpha" or "C". When fitting, the PPM class will select this hyperparameter
    using efficient approximate leave-one-out calculations.

    Parameters
    ----------
    estimator: scikit estimator object
        The regression or classification model used to obtain predictions.
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    inv_link_fn: function
        The inverse of the GLM link function.
    l_dot: function
        The first derivative of the log likelihood (with respect to the linear
        predictor), as a function of the linear predictor and the response y.
    l_doubledot: function
        The second derivative of the log likelihood (with respect to the linear
        predictor), as a function of the linear predictor and the true response
        y.
    r_doubledot: function
        The second derivative of the regularizer with respect to each
        coefficient. We assume that the regularizer is separable and symmetric
        with respect to the coefficients.
    hyperparameter_scorer: function
        The function used to evaluate different hyperparameter values.
        Typically, this is the loglikelihood as a function of the linear
        predictor and the true response y.
    trim: float
        The amount by which to trim predicted probabilities away from 0 and 1.
        This helps to stabilize some loss calculations.
    gcv_mode: string in {"auto", "svd", "eigen"}
        Flag indicating which strategy to use when performing leave-one-out
        cross-validation for ridge regression, if applicable.
        See gcv_mode in sklearn.linear_model.RidgeCV for details.
    cv_mode: None for LOOCV, int for k-fold CV with integer being # of folds
    """

    def __init__(self, estimator, loo=True, alpha_grid=np.logspace(-4, 4, 10),
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error,
                 trim=None, gcv_mode='auto', cv_ridge = None,
                 lambda_zero = False):
        super().__init__(estimator)
        self.loo = loo
        self.alpha_grid = alpha_grid
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.trim = trim
        self.gcv_mode = gcv_mode
        self.cv_ridge = cv_ridge
        self.lambda_zero = lambda_zero
        self.hyperparameter_scorer = hyperparameter_scorer
        self.alpha_ = {}
        self.loo_coefficients_ = {}
        self.coefficients_ = {}
        self._intercept_pred = None

    def _fit_model(self, X, y):
        y_train = copy.deepcopy(y)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        self._n_outputs = y_train.shape[1]
        for j in range(self._n_outputs):
            yj = y_train[:, j]
            # Compute regularization hyperparameter using approximate LOOCV or k-fold CV
            if isinstance(self.estimator, Ridge):
                if self.cv_ridge == 0:
                    self.alpha_[j] = 0.000001
                else:
                    cv = RidgeCV(alphas = self.alpha_grid, gcv_mode=self.gcv_mode,
                                cv=self.cv_ridge)
                    cv.fit(X, yj)
                    self.alpha_[j] = cv.alpha_
            else:
                self.alpha_[j] = self._get_aloocv_alpha(X, yj)
            # Fit the model on the training set and compute the coefficients
            if self.loo:
                self.loo_coefficients_[j] = \
                    self._fit_loo_coefficients(X, yj, self.alpha_[j])
                self.coefficients_[j] = _extract_coef_and_intercept(self.estimator)
            else:
                self.coefficients_[j] = \
                    self._fit_coefficients(X, yj, self.alpha_[j])

    def predict(self, X):
        preds_list = []
        for j in range(self._n_outputs):
            preds_j = _get_preds(X, self.coefficients_[j], self.inv_link_fn)
            preds_list.append(preds_j)
        if self._n_outputs == 1:
            preds = preds_list[0]
        else:
            preds = np.stack(preds_list, axis=1)
        return _trim_values(preds, self.trim)

    def predict_loo(self, X):
        preds_list = []
        for j in range(self._n_outputs):
            if self.loo:
                preds_j = _get_preds(X, self.loo_coefficients_[j], self.inv_link_fn)
            else:
                preds_j = _get_preds(X, self.coefficients_[j], self.inv_link_fn)
            preds_list.append(preds_j)
        if self._n_outputs == 1:
            preds = preds_list[0]
        else:
            preds = np.stack(preds_list, axis=1)
        return _trim_values(preds, self.trim)

    def predict_full(self, blocked_data):
        return self.predict_loo(blocked_data.get_all_data())

    def predict_partial_k(self, blocked_data, k, mode, zero_value=None):
        assert mode in ["keep_k", "keep_rest"]
        if mode == "keep_k":
            block_indices = blocked_data.get_block_indices(k)
            data_block = blocked_data.get_block(k)
        elif mode == "keep_rest":
            block_indices = blocked_data.get_all_except_block_indices(k)
            data_block = blocked_data.get_all_except_block(k)
        if len(block_indices) == 0:  # If empty block
            return self.intercept_pred
        else:
            # print("N_OUTPUTS: ", self._n_outputs)
            # SET ALL COLUMNS/STUMPS NOT FALLING IN SAMPLE'S DECISION PATH EQUAL TO COLUMN MEAN
            if zero_value is not None:
                for idx in range(len(zero_value)):
                    # print("IN HEYDAY ZONE")
                    # replace any observations in the idx-th column of data block which are equl to zero_value[idx]
                    # with the mean of the idx-th column of data block
                    data_block[:, idx] = np.where(data_block[:, idx] == zero_value[idx],
                                                    np.mean(data_block[:, idx]), data_block[:, idx])
                    
            partial_preds_list = []
            for j in range(self._n_outputs):
                if self.loo:
                    coefs = self.loo_coefficients_[j][:, block_indices]
                    intercept = self.loo_coefficients_[j][:, -1]
                else:
                    coefs = self.coefficients_[j][block_indices]
                    intercept = self.coefficients_[j][-1]
                partial_preds_j = _get_preds(
                    data_block, coefs, self.inv_link_fn, intercept
                )
                partial_preds_list.append(partial_preds_j)
            if self._n_outputs == 1:
                partial_preds = partial_preds_list[0]
            else:
                partial_preds = np.stack(partial_preds_list, axis=1)
            return _trim_values(partial_preds, self.trim)

    @property
    def intercept_pred(self):
        # print("IN 'intercept_pred' method of _GlmPPM")
        if self._intercept_pred is None:
            self._intercept_pred = np.array([
                _trim_values(self.inv_link_fn(self.coefficients_[j][-1]), self.trim) \
                for j in range(self._n_outputs)
            ])
        return ("constant_model", self._intercept_pred)

    def _fit_coefficients(self, X, y, alpha):
        # print("IN '_fit_coefficients' method of _GlmPPM")
        _set_alpha(self.estimator, alpha)
        self.estimator.fit(X, y)
        return _extract_coef_and_intercept(self.estimator)

    def _fit_loo_coefficients(self, X, y, alpha, max_h=1-1e-4):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """
        orig_coef_ = self._fit_coefficients(X, y, alpha)
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
        support_idxs = orig_coef_ != 0
        if not any(support_idxs):
            return orig_coef_ * np.ones_like(X1)
        X1 = X1[:, support_idxs]
        orig_coef_ = orig_coef_[support_idxs]
        l_doubledot_vals = self.l_doubledot(y, orig_preds)
        J = X1.T * l_doubledot_vals @ X1
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(orig_coef_) * \
                               np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0 # Do not penalize constant term
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        normal_eqn_mat = np.linalg.inv(J) @ X1.T
        h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
        h_vals[h_vals == 1] = max_h
        loo_coef_ = orig_coef_[:, np.newaxis] + \
                    normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
        if not all(support_idxs):
            loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            loo_coef_dense_[support_idxs, :] = loo_coef_
            loo_coef_ = loo_coef_dense_
        return loo_coef_.T

    def _get_aloocv_alpha(self, X, y):
        cv_scores = np.zeros_like(self.alpha_grid)
        for i, alpha in enumerate(self.alpha_grid):
            loo_coef_ = self._fit_loo_coefficients(X, y, alpha)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            sample_scores = np.sum(loo_coef_ * X1, axis=1)
            preds = _trim_values(self.inv_link_fn(sample_scores), self.trim)
            cv_scores[i] = self.hyperparameter_scorer(y, preds)
        return self.alpha_grid[np.argmin(cv_scores)]


class GlmRegressorPPM(_GlmPPM, PartialPredictionModelBase, ABC):
    """
    PPM class for GLM regression estimator.
    """
    ...


class GlmClassifierPPM(_GlmPPM, PartialPredictionModelBase, ABC):
    """
    PPM class for GLM classification estimator.
    """

    def predict_proba(self, X):
        # print("IN 'predict_proba' method of GlmClassifierPPM")
        probs = self.predict(X)
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        return probs

    def predict_proba_loo(self, X):
        # print("IN 'predict_proba_loo' method of GlmClassifierPPM")
        probs = self.predict_loo(X)
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        return probs


class _RidgePPM(_GlmPPM, PartialPredictionModelBase, ABC):
    """
    PPM class that uses ridge as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    gcv_mode: string in {"auto", "svd", "eigen"}
        Flag indicating which strategy to use when performing leave-one-out
        cross-validation for ridge regression.
        See gcv_mode in sklearn.linear_model.RidgeCV for details.
    **kwargs
        Other Parameters are passed on to Ridge().
    """

    def __init__(self, loo=True, alpha_grid=np.logspace(-5, 5, 100),
                 gcv_mode='auto', cv_ridge=None, **kwargs):
        # print("CREATING _RidgePPM OBJECT")
        super().__init__(Ridge(**kwargs), loo, alpha_grid, gcv_mode=gcv_mode,
                         cv_ridge = cv_ridge)

    def set_alphas(self, alphas="default", blocked_data=None, y=None):
        # print("IN 'set_alphas' method of _RidgePPM")
        full_data = blocked_data.get_all_data()
        if alphas == "default":
            alphas = get_alpha_grid(full_data, y)
        else:
            alphas = alphas
        self.alpha_grid = alphas


class RidgeRegressorPPM(_RidgePPM, GlmRegressorPPM,
                        PartialPredictionModelBase, ABC):
    """
    PPM class for regression that uses ridge as the GLM estimator.
    """
    ...


class RidgeClassifierPPM(_RidgePPM, GlmClassifierPPM,
                         PartialPredictionModelBase, ABC):
    """
    PPM class for classification that uses ridge as the GLM estimator.
    """

    def predict_proba(self, X):
        # print("IN 'predict_proba' method of RidgeClassifierPPM")
        probs = softmax(self.predict(X))
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        return probs

    def predict_proba_loo(self, X):
        # print("IN 'predict_proba_loo' method of RidgeClassifierPPM")
        probs = softmax(self.predict_loo(X))
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        return probs


class LogisticClassifierPPM(GlmClassifierPPM, PartialPredictionModelBase, ABC):
    """
    PPM class for classification that uses logistic regression as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    max_iter: int
        The maximum number of iterations for the LogisticRegression solver.
    trim: float
        The amount by which to trim predicted probabilities away from 0 and 1.
        This helps to stabilize some loss calculations.
    **kwargs
        Other Parameters are passed on to LogisticRegression().
    """

    def __init__(self, loo=True, alpha_grid=np.logspace(-2, 3, 25),
                 penalty='l2', max_iter=1000, trim=0.01, **kwargs):
        # print("CREATING LogisticClassifierPPM OBJECT")
        assert penalty in ['l2', 'l1']
        if penalty == 'l2':
            r_doubledot = lambda a: 1
        elif penalty == 'l1':
            r_doubledot = None
        super().__init__(LogisticRegression(penalty=penalty, max_iter=max_iter, **kwargs),
                         loo, alpha_grid,
                         inv_link_fn=sp.special.expit,
                         l_doubledot=lambda a, b: b * (1 - b),
                         r_doubledot=r_doubledot,
                         hyperparameter_scorer=log_loss,
                         trim=trim)


class RobustRegressorPPM(GlmRegressorPPM, PartialPredictionModelBase, ABC):
    """
    PPM class for regression that uses Huber robust regression as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    epsilon: float
        The robustness parameter for Huber regression. The smaller the epsilon,
        the more robust it is to outliers. Epsilon must be in the range
        [1, inf).
    **kwargs
        Other Parameters are passed on to LogisticRegression().
    """
    def __init__(self, loo=True, alpha_grid=np.logspace(-2, 3, 25),
                 epsilon=1.35, max_iter=2000, **kwargs):
        # print("CREATING RobustRegressorPPM OBJECT")
        loss_fn = partial(huber_loss, epsilon=epsilon)
        l_dot = lambda a, b: (b - a) / (1 + ((a - b) / epsilon) ** 2) ** 0.5
        l_doubledot=lambda a, b: (1 + (((a - b) / epsilon) ** 2)) ** (-1.5)
        super().__init__(
            HuberRegressor(max_iter=max_iter, **kwargs), loo, alpha_grid,
            l_dot=l_dot,
            l_doubledot=l_doubledot,
            hyperparameter_scorer=loss_fn)


class LassoRegressorPPM(GlmRegressorPPM, PartialPredictionModelBase, ABC):
    """
    PPM class for regression that uses lasso as the estimator.

    Parameters
    ----------
    loo: bool
        Flag for whether to also use LOO calculations for making predictions.
    alpha_grid: ndarray of shape (n_alphas, )
        The grid of alpha values for hyperparameter optimization.
    **kwargs
        Other Parameters are passed on to Lasso().
    """

    def __init__(self, loo=True, alpha_grid=np.logspace(-2, 3, 25), **kwargs):
        # print("CREATING LassoRegressorPPM OBJECT")
        super().__init__(Lasso(**kwargs), loo, alpha_grid, r_doubledot=None)

