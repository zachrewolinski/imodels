import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score, log_loss
from scipy.special import softmax
from scipy import sparse
from glmnet import LogitNet
warnings.filterwarnings("ignore")
from scipy.special import expit, logit

from imodels.tree.rf_plus.ppms.ppms import PartialPredictionModelBase, _GlmPPM

from imodels.tree.rf_plus.ppms.ppms_util import _extract_coef_and_intercept, _get_preds



class GlmClassifierPPM(PartialPredictionModelBase, ABC):
    """
    PPM class for GLM regression estimator based on the glmnet implementation. Only can deal with binary for now. 
    """
    def __init__(self, estimator, cv="loo", n_alphas=100, l1_ratio=0.5,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a,
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1, sample_weights = 'balanced',
                 hyperparameter_scorer=log_loss):
        
        super().__init__(estimator)
        self.cv = cv
        self.n_alphas = n_alphas
        self.inv_link_fn = inv_link_fn
        self.l_dot = l_dot
        self.l_doubledot = l_doubledot
        self.r_doubledot = r_doubledot
        self.hyperparameter_scorer = hyperparameter_scorer
        self._augmented_coeffs_for_each_alpha = {} #coefficients and intercept for all reg params
        self.alpha_ = {}
        self.loo_coefficients_ = None
        self.coefficients_ = None
        self._intercept_pred = None
        self.l1_ratio = l1_ratio
        self.sample_weights = sample_weights
        
    def _fit_model(self, X, y):

        y_train = copy.deepcopy(y)
        if self.sample_weights == 'balanced':
            n_pos = np.sum(y_train)
            n_neg = len(y_train) - n_pos
            weight_pos = (1 / n_pos) * (n_pos + n_neg) / 2.0
            weight_neg = (1 / n_neg) * (n_pos + n_neg) / 2.0
            self.sample_weights = np.where(y_train == 1, weight_pos, weight_neg) * len(y_train)
        else:
            self.sample_weights = np.ones(len(y_train))/len(y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.estimator.fit(X, y_train, sample_weight=self.sample_weights)   
        
        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._augmented_coeffs_for_each_alpha[lambda_] = np.hstack([self.estimator.coef_path_[0,:, i],self.estimator.intercept_path_[0,i]]) #only binary classification        

        #compute regularization hyperparameter using approximate LOOCV or k-fold CV
        if self.cv == "loo":
            self.alpha_ = self._get_aloocv_alpha(X, y)
        else:
            self.alpha_ = self.estimator.lambda_best_[0]

        #fit the model on the training set and compute the coefficients
        if self.cv == "loo":
            self.loo_coefficients_ = self._fit_loo_coefficients(X, y_train, self.alpha_)
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
        else:
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]

    def _get_aloocv_alpha(self, X, y):
        cv_scores = np.zeros(len(self.estimator.lambda_path_))
        for i, lambda_ in enumerate(self.estimator.lambda_path_):
            loo_coef_ = self._fit_loo_coefficients(X, y, lambda_)
            X1 = np.hstack([X, np.ones((X.shape[0], 1))])
            sample_scores = np.sum(loo_coef_ * X1, axis=1)
            preds = self.inv_link_fn(sample_scores)
            cv_scores[i] = self.hyperparameter_scorer(y, preds,sample_weight=self.sample_weights)
        return self.estimator.lambda_path_[np.argmin(cv_scores)]

    def _fit_loo_coefficients(self, X, y, lambda_, max_h=1-1e-4):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """
        orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
        support_idxs = orig_coef_ != 0
        if not any(support_idxs):
            return orig_coef_ * np.ones_like(X1)
        X1 = X1[:, support_idxs]
        orig_coef_ = orig_coef_[support_idxs]
        l_doubledot_vals = self.l_doubledot(y, orig_preds) * self.sample_weights
        J = X1.T * l_doubledot_vals @ X1
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0
            reg_curvature = np.diag(r_doubledot_vals)
            J += lambda_ * reg_curvature
        normal_eqn_mat = np.linalg.inv(J) @ X1.T
        h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
        h_vals[h_vals == 1] = max_h
        l_dot_vals = self.l_dot(y, orig_preds) * self.sample_weights
        loo_coef_ = orig_coef_[:, np.newaxis] + normal_eqn_mat * l_dot_vals / (1 - h_vals)
        if not all(support_idxs):
            loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            loo_coef_dense_[support_idxs, :] = loo_coef_
            loo_coef_ = loo_coef_dense_
        return loo_coef_.T

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.stack([1 - preds, preds]).T

    def predict(self, X):
        preds = _get_preds(X, self.coefficients_, self.inv_link_fn)
        return preds
    
      
    def predict_loo(self, X):
        if self.cv == "loo":
            return _get_preds(X, self.loo_coefficients_, self.inv_link_fn)
            
        else:
            return _get_preds(X, self.coefficients_, self.inv_link_fn)
         

    def predict_full(self, blocked_data):
        return self.predict(blocked_data.get_all_data())

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
            if zero_value is not None:
                for idx in range(len(zero_value)):
                    # replace any observations in the idx-th column of data block which are equl to zero_value[idx] with the mean of the idx-th column of data block
                    data_block[:, idx] = np.where(data_block[:, idx] == zero_value[idx],np.mean(data_block[:, idx]), data_block[:, idx])
        if self.loo == "loo":
            coefs = self.loo_coefficients_[:,block_indices]
            intercept = self.loo_coefficients_[:,-1]
        else:
            coefs = self.coefficients_[block_indices]
            intercept = self.coefficients_[-1]
        partial_preds_k = _get_preds(data_block, coefs, self.inv_link_fn, intercept)
        return partial_preds_k
  
    @property
    def intercept_pred(self):
        if self._intercept_pred is None:
            self._intercept_pred = np.array([self.inv_link_fn(self.coefficients_[-1])])
        return ("constant_model", self._intercept_pred)



class GLMLogisticElasticNetPPM(GlmClassifierPPM):
    """
    PPM class for Logistic Regression with Elastic Net Penalty
    """

    def __init__(self, cv="loo", n_alphas=100, l1_ratio = 0.5, standardize = False, sample_weights = 'balanced',**kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = l1_ratio,standardize = standardize, **kwargs), 
                        cv,inv_link_fn=sp.special.expit, l_dot=lambda a, b: b - a, l1_ratio=l1_ratio, sample_weights = sample_weights,
                        l_doubledot=lambda a, b:  b * (1-b), r_doubledot=lambda a: 1 * (1 - l1_ratio),
                        hyperparameter_scorer=log_loss)
        
class GLMLogisticLassoNetPPM(GlmClassifierPPM):
    """
    PPM class for Logistic Regression with Lasso Penalty
    """

    def __init__(self, cv="loo", n_alphas=100, standardize = False, sample_weights = 'balanced',**kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 1,standardize = standardize, **kwargs), 
                        cv,inv_link_fn=sp.special.expit, l_dot=lambda a, b: b - a, l1_ratio=1, sample_weights = sample_weights,
                        l_doubledot=lambda a, b:  b * (1-b), r_doubledot=None,
                        hyperparameter_scorer=log_loss)
        
class GLMLogisticRidgeNetPPM(GlmClassifierPPM):
    """
    PPM class for Logistic Regression with Lasso Penalty
    """

    def __init__(self, cv="loo", n_alphas=100, standardize = False, sample_weights = 'balanced',**kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 0,standardize = standardize, **kwargs), 
                        cv,inv_link_fn=sp.special.expit, l_dot=lambda a, b: b - a, l1_ratio=0, sample_weights = sample_weights,
                        l_doubledot=lambda a, b:  b * (1-b), r_doubledot=None,
                        hyperparameter_scorer=log_loss)



class GLMSVMPPM(GlmClassifierPPM):
    pass 

if __name__ == "__main__":
    # from sklearn.datasets import make_regression
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    

    model = GLMLogisticElasticNetPPM(l1_ratio=0.5)
    model.fit(X_train, y_train)

    model = GLMLogisticLassoNetPPM()
    model.fit(X_train, y_train)

    model = GLMLogisticRidgeNetPPM()
    model.fit(X_train, y_train)
    print(roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))    
