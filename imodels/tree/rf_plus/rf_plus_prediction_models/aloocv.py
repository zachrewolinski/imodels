# Generic Imports 
import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import time, numbers
import numpy as np
import scipy as sp
import pandas as pd
from collections import OrderedDict


# Sklearn Imports
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import  mean_squared_error

#scipy imports
from scipy.special import softmax
from scipy import linalg



class AloGLM():
    """
    Predictive Linear Model with approximate leave-one-out cross-validation
    """
    def __init__(self, estimator, cv="loo", n_alphas=100, l1_ratio=0.0, standardize = False,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=mean_squared_error):
        
        estimator.__init__()
        self.estimator = estimator  
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
        self.influence_matrix_ = None
        self.support_idxs = None
        self.standardize = standardize
        
    def fit(self, X, y):
        y_train = copy.deepcopy(y)
        self.estimator.fit(X, y_train)

        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._augmented_coeffs_for_each_alpha[lambda_] = np.hstack([self.estimator.coef_path_[:, i],self.estimator.intercept_path_[i]])         

        if self.cv == "loo":   
           self._get_aloocv_alpha(X, y_train)  
           self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
           self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y,self.coefficients_,self.alpha_)
           self.support_idxs_ = np.where(self.coefficients_ != 0)[0]

        else:
            self.alpha_ = self.estimator.lambda_max_[0]
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]

    
    def _get_aloocv_alpha(self, X, y,max_h = 1 - 1e-5):
        #Assume we are solving 1/n l_i + lambda * r
       
        all_support_idxs = {}
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X1.shape[0]
        best_lambda_ = -1
        best_cv_ = np.inf
 
        #get min of self.estimator.lambda_path
        min_lambda_ = np.min(self.estimator.lambda_path_)

        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
            support_idxs_lambda_ = orig_coef_ != 0
            X1_support = X1[:, support_idxs_lambda_]
            
            if tuple(support_idxs_lambda_) not in all_support_idxs:

                u,s,vh =  linalg.svd(X1_support,full_matrices=False)  
                us = u * s
                ush = us.T
                all_support_idxs[tuple(support_idxs_lambda_)] = s,us,ush
            else:
                s,us,ush = all_support_idxs[tuple(support_idxs_lambda_)]
            
            orig_preds = self._get_preds(X, orig_coef_, self.inv_link_fn)
            l_doubledot_vals = self.l_doubledot(y, orig_preds)/n 
            orig_coef_ = orig_coef_[np.array(support_idxs_lambda_)]
            
            if self.r_doubledot is not None:
                r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
                r_doubledot_vals[-1] = 0
                reg_curvature =  lambda_ * r_doubledot_vals
            else: 
                r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_) 
                r_doubledot_vals[-1] = 0
                reg_curvature =  min_lambda_ * r_doubledot_vals
            
            if l_doubledot_vals is isinstance(l_doubledot_vals, float):
                diag_elements = s * l_doubledot_vals * s
            else:
                diag_elements = s * l_doubledot_vals[:len(s)] * s

            
            Sigma2_plus_lambda = diag_elements + reg_curvature
            Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda)
            
            h_vals = np.einsum('ij,j,jk->i', us,Sigma2_plus_lambda_inverse,ush,optimize=True) * l_doubledot_vals
            h_vals[h_vals == 1] = max_h
            l_dot_vals = self.l_dot(y, orig_preds) / n
            loo_preds = orig_preds + h_vals * l_dot_vals / (1 - h_vals)
            sample_scores = self.hyperparameter_scorer(y, loo_preds)
            if sample_scores < best_cv_:
                best_cv_ = sample_scores
                best_lambda_ = lambda_

        self.alpha_ = best_lambda_
                                                                              
    
    def _get_loo_coefficients(self,X, y, orig_coef_,alpha,max_h=1-1e-4):
        """
        Get the coefficient (and intercept) for each LOO model. Since we fit
        one model for each sample, this gives an ndarray of shape (n_samples,
        n_features + 1)
        """

        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        n = X.shape[0]
        orig_preds = self._get_preds(X, orig_coef_, self.inv_link_fn)   #self._get_preds(X, orig_coef_, inv_link_fn)
        support_idxs = orig_coef_ != 0
        if not any(support_idxs):
            return orig_coef_ * np.ones_like(X1)
        X1 = X1[:, support_idxs]
        orig_coef_ = orig_coef_[support_idxs]
        l_doubledot_vals = self.l_doubledot(y, orig_preds)/n
        J = X1.T * l_doubledot_vals @ X1
        if self.r_doubledot is not None:
            r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
            r_doubledot_vals[-1] = 0 # Do not penalize constant term
            reg_curvature = np.diag(r_doubledot_vals)
            J += alpha * reg_curvature
        
        normal_eqn_mat = np.linalg.inv(J) @ X1.T
        
        h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals/n
        h_vals[h_vals == 1] = max_h
        l_dot_vals = self.l_dot(y, orig_preds)/n
        influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)

        loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix

        if not all(support_idxs):
            loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            loo_coef_dense_[support_idxs, :] = loo_coef_
            loo_coef_ = loo_coef_dense_
        return loo_coef_.T,influence_matrix

    
    def predict(self, X):
        preds = self._get_preds(X, self.coefficients_, self.inv_link_fn)
        return preds
    
      
    def predict_loo(self, X):
        if self.cv == "loo":
            return self._get_preds(X, self.loo_coefficients_, self.inv_link_fn)
            
        else:
            return self._get_preds(X, self.coefficients_, self.inv_link_fn)
    

    def _get_preds(self,X, coefs, inv_link_fn, intercept=None):
        if coefs.ndim > 1: # LOO predictions
            if coefs.shape[1] == (X.shape[1] + 1):
                intercept = coefs[:, -1]
                coefs = coefs[:, :-1]
            lin_preds = np.sum(X * coefs, axis=1) + intercept
        else:
            if len(coefs) == (X.shape[1] + 1):
                intercept = coefs[-1]
                coefs = coefs[:-1]           
            lin_preds = X @ coefs + intercept
        return inv_link_fn(lin_preds)
    

    @property
    def intercept_pred(self):
        if self._intercept_pred is None:
            self._intercept_pred = np.array([self.inv_link_fn(self.coefficients_[-1])])
        return ("constant_model", self._intercept_pred)


