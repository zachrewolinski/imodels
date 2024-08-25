#Generic imports
import copy
from abc import ABC, abstractmethod
import warnings
from functools import partial
import numpy as np
import scipy as sp
import pandas as pd
from scipy.special import softmax, expit
import time
from tqdm import tqdm

# Sklearn imports
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

# imodels/RFPlus imports
import imodels 
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloGLMRegressor, AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.data_transformations.block_transformers import MDIPlusDefaultTransformer, TreeTransformer, CompositeTransformer, IdentityTransformer



class _MDIPlusPartialPredictionModelBase(ABC):
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
    assumes it is fitted.
    """

    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)
    

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

class _MDIPlusGenericPPM(_MDIPlusPartialPredictionModelBase, ABC):
    """
    Partial prediction model for arbitrary estimators. May be slow.
    """

    def __init__(self, estimator):
        super().__init__(estimator)

    def predict_full(self, blocked_data):
        return self.estimator.predict(blocked_data.get_all_data())

    def predict_partial_k(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict(modified_data)
    
    def predict_partial_subtract_intercept(self, blocked_data, mode, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode)
        return partial_preds

    def predict_partial_subtract_constant(self, blocked_data, constant, mode, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k_subtract_constant(blocked_data, k, mode, constant, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k_subtract_constant(blocked_data, k, mode, constant)
        return partial_preds



class MDIPlusGenericRegressorPPM(_MDIPlusGenericPPM, ABC):
    """
    Partial prediction model for arbitrary regression estimators. May be slow.
    """
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict(modified_data) - self.estimator.intercept_
    
    def predict_partial_k_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict(modified_data) - constant


class MDIPlusGenericClassifierPPM(_MDIPlusGenericPPM, ABC):
    """
    Partial prediction model for arbitrary classification estimators. May be slow.
    """

    def predict_full(self, blocked_data):
        return self.estimator.predict_proba(blocked_data.get_all_data())[:,1]

    def predict_partial_k(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_proba(modified_data)[:,1]
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - self.estimator.intercept_

    def predict_partial_k_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - constant

class AloMDIPlusPartialPredictionModelRegressor(_MDIPlusGenericPPM,AloGLMRegressor):
    '''
    Assumes that the estimator has a loo_coefficients_ attribute.
    '''
    
    def predict_partial_k_loo(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_loo(modified_data)
   
    def predict_partial_loo(self, blocked_data, mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode)
        return partial_preds
    
    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_loo(blocked_data.get_all_data())
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict(modified_data) - self.estimator.intercept_
    
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_loo(modified_data) - self.estimator.intercept_
    
    def predict_partial_loo_subtract_intercept(self, blocked_data, mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode)
        return partial_preds
    
    def predict_partial_k_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict(modified_data) - constant
    
    def predict_partial_k_loo_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_loo(modified_data) - constant
    
    def predict_partial_loo_subtract_constant(self, blocked_data, constant, mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_constant(blocked_data, k, mode, constant)
        return partial_preds

class AloMDIPlusPartialPredictionModelClassifier(_MDIPlusGenericPPM,AloGLMClassifier):
    
    '''
    Assumes that the estimator has a loo_coefficients_ attribute.
    '''
    def predict_partial(self, blocked_data,mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k(blocked_data, k, mode)
        return partial_preds
    
    def predict_partial_k(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_proba(modified_data)[:,1]
    
    def predict_partial_loo(self, blocked_data,mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode)
        return partial_preds
    
    def predict_partial_k_loo(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        return self.estimator.predict_proba_loo(modified_data)[:,1]
    
    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_proba_loo(blocked_data.get_all_data())[:,1]
    
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba_loo(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - self.estimator.intercept_
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - self.estimator.intercept_

    def predict_partial_loo_subtract_intercept(self, blocked_data, mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode)
        return partial_preds
    
    def predict_partial_k_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - constant
    
    def predict_partial_k_loo_subtract_constant(self, blocked_data, k, mode, constant):
        modified_data = blocked_data.get_modified_data(k, mode)
        probabilities = self.estimator.predict_proba_loo(modified_data)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities))
        return log_odds - constant
    
    def predict_partial_loo_subtract_constant(self, blocked_data, constant, mode):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_constant(blocked_data, k, mode, constant)
        return partial_preds

    

if __name__ == "__main__":
    
    # Test MDIPlus Regression
    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    rfplus_reg = RandomForestPlusRegressor()
    rfplus_reg.fit(X_train, y_train)

   
    for i in tqdm(range(len(rfplus_reg.estimators_))):
        mdiplus_ppm = AloMDIPlusPartialPredictionModelRegressor(rfplus_reg.estimators_[i])
        train_blocked_data_i = rfplus_reg.transformers_[i].transform(X_train)
        test_blocked_data_i = rfplus_reg.transformers_[i].transform(X_test)
        partial_loo_preds = mdiplus_ppm.predict_partial_loo(train_blocked_data_i,mode="keep_k")
        partial_preds = mdiplus_ppm.predict_partial(test_blocked_data_i,mode="keep_k")
        print(f"Partial preds number of features: {len(partial_preds)}")
        print(f"Partial preds number of samples per feature: {len(partial_preds[0])}")

        print(f"LOO Partial preds number of features: {len(partial_loo_preds)}")
        print(f"LOO Partial preds number of samples per feature: {len(partial_loo_preds[0])}")
        break
    

    # Test MDIPlus Classification
    # X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    # rfplus_clf = RandomForestPlusClassifier()
    # rfplus_clf.fit(X_train, y_train)

    # print(X_train.shape)

    # print("Done fitting rfplus classifier")
   
    # for i in range(len(rfplus_clf.estimators_)):
    #     mdiplus_ppm_clf = AloMDIPlusPartialPredictionModelRegressor(rfplus_clf.estimators_[i])
    #     train_blocked_data_i = rfplus_clf.transformers_[i].transform(X_train)
    #     test_blocked_data_i = rfplus_clf.transformers_[i].transform(X_test)
    #     partial_loo_preds = mdiplus_ppm_clf.predict_partial_loo(train_blocked_data_i,mode="keep_k")
    #     partial_preds = mdiplus_ppm_clf.predict_partial(test_blocked_data_i,mode="keep_k")
    #     print(f"Partial preds: {partial_preds}")
    #     break








    
    
    
    
    
    
    # def _fit_model(self, X, y):
    #     self.estimator.fit(X, y)

    # def predict(self, X):
    #     return self.estimator.predict(X)
    
    # @abstractmethod
    # def predict(self, X):
    #     """
    #     Make predictions on new data using the fitted model.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix, for which to make predictions.
    #     """
    #     pass

    #self.is_fitted = False

    # def fit(self, X, y):
    #     """
    #     Fit the partial prediction model.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix.
    #     y: ndarray of shape (n_samples, n_targets)
    #         The observed responses.
    #     """
    #     self._fit_model(X, y)
    #     self.is_fitted = True

    # @abstractmethod
    # def _fit_model(self, X, y):
    #     """
    #     Fit the regression or classification model on all the data.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix.
    #     y: ndarray of shape (n_samples, n_targets)
    #         The observed responses.
    #     """
    #    pass