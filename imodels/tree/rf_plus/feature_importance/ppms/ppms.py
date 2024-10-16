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
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM


class MDIPlusGenericRegressorPPM(ABC):
    """
    Partial prediction model for arbitrary estimators. May be slow.
    """

    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        return self.estimator.predict(blocked_data.get_all_data())
    
    def predict_partial(self, blocked_data, mode, l2norm, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode, l2norm, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode, l2norm)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, mode, l2norm, sign, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode, l2norm=l2norm, sign=sign, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode, l2norm=l2norm, sign=sign)
        return partial_preds
    
    def predict_partial_k(self, blocked_data, k, mode, l2norm):
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            if isinstance(self.estimator, AloGLM):
                coefs = self.estimator.coefficients_
            else:
                coefs = self.estimator.coef_
            return ((modified_data**2) @ (coefs**2))**(1/2) + self.estimator.intercept_
        return self.estimator.predict(modified_data)

    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode, l2norm, sign):
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            if isinstance(self.estimator, AloGLM):
                coefs = self.estimator.coefficients_
            else:
                coefs = self.estimator.coef_
            if sign:
                sign_term = np.sign(modified_data @ coefs)
                return sign_term * ((modified_data**2) @ (coefs**2))**(1/2)
            return ((modified_data**2) @ (coefs**2))**(1/2)
        return self.estimator.predict(modified_data) - self.estimator.intercept_


class MDIPlusGenericClassifierPPM(ABC):
    """
    Partial prediction model for arbitrary classification estimators. May be slow.
    """
    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        return self.estimator.predict_proba(blocked_data.get_all_data())[:,1]
    
    def predict_partial(self, blocked_data, mode, l2norm, sigmoid=False, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode, l2norm, sigmoid, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k(blocked_data, k, mode, l2norm, sigmoid)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, mode, l2norm, sign, sigmoid=False, zero_values=None):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            if zero_values is not None:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode, l2norm=l2norm, sigmoid=sigmoid, sign=sign, zero_value=zero_values[k])
            else:
                partial_preds[k] = self.predict_partial_k_subtract_intercept(blocked_data, k, mode, l2norm=l2norm, sigmoid=sigmoid,  sign=sign)
        return partial_preds
    

    def predict_partial_k(self, blocked_data, k, mode, l2norm, sigmoid):
        modified_data = blocked_data.get_modified_data(k, mode)
        if isinstance(self.estimator, AloGLM):
            coefs = self.estimator.coefficients_
        else:
            coefs = self.estimator.coef_
        if l2norm:
            if sigmoid:
                return expit(((modified_data**2) @ (coefs**2))**(1/2) + self.estimator.intercept_)
            return ((modified_data**2) @ (coefs**2))**(1/2) + self.estimator.intercept_
        if sigmoid:
            return self.estimator.predict_proba(modified_data)[:,1]
        return modified_data @ coefs + self.estimator.intercept_
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode, l2norm, sigmoid, sign):
        modified_data = blocked_data.get_modified_data(k, mode)
        if isinstance(self.estimator, AloGLM):
            coefs = self.estimator.coefficients_
        else:
            coefs = self.estimator.coef_
        if l2norm:
            if sigmoid:
                return expit(((modified_data**2) @ (coefs**2))**(1/2))
            if sign:
                sign_term = np.sign(modified_data @ coefs)
                return sign_term*((modified_data**2) @ (coefs**2))**(1/2)
            return ((modified_data**2) @ (coefs**2))**(1/2)
        if sigmoid:
            return  expit(modified_data @ coefs)
        return modified_data @ coefs


class AloMDIPlusPartialPredictionModelRegressor(MDIPlusGenericRegressorPPM,AloGLMRegressor):
    """
    Assumes that the estimator has a loo_coefficients_ attribute.
    """

    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_loo(blocked_data.get_all_data())

    def predict_partial_loo(self, blocked_data, mode, l2norm):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode, l2norm)
        return partial_preds

    def predict_partial_loo_subtract_intercept(self, blocked_data, mode, l2norm, sign):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode, l2norm, sign=sign)
        return partial_preds

    def predict_partial_k_loo(self, blocked_data, k, mode, l2norm):
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            coefs = self.estimator.loo_coefficients_[:, :-1] 
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1]
        return self.estimator.predict_loo(modified_data)
   
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode, l2norm, sign):
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            coefs = self.estimator.loo_coefficients_[:, :-1]
            if sign:
                signs = np.sign(np.sum(modified_data * coefs, axis=1))
                return signs*np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
        return self.estimator.predict_loo(modified_data) - self.estimator.loo_coefficients_[:, -1]
    

class AloMDIPlusPartialPredictionModelClassifier(MDIPlusGenericClassifierPPM,AloGLMClassifier):
    """
    Assumes that the estimator has a loo_coefficients_ attribute.
    """    
    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_proba_loo(blocked_data.get_all_data())[:,1]

    def predict_partial_loo(self, blocked_data, mode, l2norm, sigmoid):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode, l2norm, sigmoid)
        return partial_preds
    
    def predict_partial_loo_subtract_intercept(self, blocked_data, mode, l2norm, sigmoid, sign):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode, l2norm, sigmoid, sign=sign)
        return partial_preds

    def predict_partial_k_loo(self, blocked_data, k, mode, l2norm, sigmoid):
        modified_data = blocked_data.get_modified_data(k, mode)
        coefs = self.estimator.loo_coefficients_[:, :-1]
        if l2norm:
            if sigmoid:
                return expit(np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1])
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1]
        if sigmoid:
            return self.estimator.predict_proba_loo(modified_data)[:,1]
        return np.sum((modified_data * coefs), axis = 1) + self.estimator.loo_coefficients_[:, -1]
    
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode, l2norm, sigmoid, sign):
        modified_data = blocked_data.get_modified_data(k, mode)
        coefs = self.estimator.loo_coefficients_[:, :-1]
        if l2norm:
            if sigmoid:
                return expit(np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1))
            if sign:
                signs = np.sign(np.sum(modified_data * coefs, axis=1))
                return signs*np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
        if sigmoid:
            return expit(np.sum(modified_data * coefs, axis = 1))
        return np.sum(modified_data * coefs, axis = 1)

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
        partial_loo_preds = mdiplus_ppm.predict_partial_loo(train_blocked_data_i,mode="keep_k",l2norm=False)
        partial_preds = mdiplus_ppm.predict_partial(test_blocked_data_i,mode="keep_k", l2norm=False)
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