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
