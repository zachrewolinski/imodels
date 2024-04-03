import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd
import time, numbers
from tqdm import tqdm
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from scipy.special import softmax
from scipy import sparse
from glmnet import ElasticNet
from scipy.sparse import csc_matrix
from scipy import linalg
from imodels.tree.rf_plus.ppms.ppms import PartialPredictionModelBase, _GlmPPM
from collections import OrderedDict
from imodels.tree.rf_plus.ppms.ppms_util import _extract_coef_and_intercept, _get_preds, _get_loo_coefficients, neg_r2_score
from sklearn.linear_model import RidgeCV


class GlmNetRegressorPPM(PartialPredictionModelBase, ABC):
    """
    PPM class for GLM regression estimator based on the glmnet implementation.
    """
    def __init__(self, estimator, cv="loo", n_alphas=100, l1_ratio=0.0, standardize = False,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=neg_r2_score):
        
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
        self.influence_matrix_ = None
        self.support_idxs = None
        self.standardize = standardize
        
    def _fit_model(self, X, y):
        y_train = copy.deepcopy(y)
        self.estimator.fit(X, y_train)

        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._augmented_coeffs_for_each_alpha[lambda_] = np.hstack([self.estimator.coef_path_[:, i],self.estimator.intercept_path_[i]])         

        if self.cv == "loo":   
           self._get_aloocv_alpha(X, y_train)  
           self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
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
            
            orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
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
            
            if l_doubledot_vals is isinstance(l_doubledot_vals, numbers.Number):
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
        self.loo_coefficients_,self.influence_matrix_ = _get_loo_coefficients(X, y,self._augmented_coeffs_for_each_alpha[self.alpha_],self.alpha_,
                                                                              self.inv_link_fn,self.l_doubledot,self.r_doubledot, self.l_dot)
    
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

    def predict_partial_k(self, blocked_data, k, mode,use_loo = True, zero_value=None):
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
        if use_loo == "loo":
            coefs = self.loo_coefficients_[:,block_indices]
            intercept = self.loo_coefficients_[:,-1]
        else:
            coefs = self.coefficients_[block_indices]
            intercept = self.coefficients_[-1]
        partial_preds_k = _get_preds(data_block, coefs, self.inv_link_fn, intercept)
        return partial_preds_k
  
    def predict_partial(self, blocked_data, mode,use_loo,zero_values=None):
       pass

    @property
    def intercept_pred(self):
        if self._intercept_pred is None:
            self._intercept_pred = np.array([self.inv_link_fn(self.coefficients_[-1])])
        return ("constant_model", self._intercept_pred)



class GlmNetElasticNetRegressorPPM(GlmNetRegressorPPM):
    """
    PPM class for Elastic
    """

    def __init__(self, cv="loo", n_alphas=100, l1_ratio = 0.5, standardize = False, **kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(ElasticNet(n_lambda=n_alphas,standardize=standardize,n_splits=n_splits,alpha=l1_ratio,**kwargs), standardize= standardize,
                        cv = cv,inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio=l1_ratio,
                        l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1 - l1_ratio,
                        hyperparameter_scorer=mean_squared_error)
        
class GlmNetRidgeRegressorPPM(GlmNetRegressorPPM):
    """
    PPM class for Elastic
    """

    def __init__(self, cv="loo", n_alphas=100, l1_ratio = 0.0,standardize = False, **kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        
        lambda_path = np.logspace(-5,5,100)
        
        super().__init__(ElasticNet(n_lambda=n_alphas,standardize=standardize,n_splits=n_splits,alpha=0.0,**kwargs), cv = cv,
                        inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio=l1_ratio,
                        l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                        hyperparameter_scorer=mean_squared_error)


class GlmNetLassoRegressorPPM(GlmNetRegressorPPM):
    """
    Ppm class for regression that uses lasso as the GLM estimator.
    """
    def __init__(self, cv="loo", n_alphas=100, standardize = False, **kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(ElasticNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 1.0,standardize = standardize, **kwargs), 
                         cv,inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, l1_ratio = 1.0,
                        l_doubledot=lambda a, b: 1, r_doubledot=None,
                        hyperparameter_scorer=mean_squared_error)
    

class SklearnRidgeRegressorPPM(GlmNetRegressorPPM):
    """
    Ppm class for regression that uses ridge as the GLM estimator.
    Uses fast scikit-learn LOO implementation
    """
    def __init__(self, cv="loo", n_alphas=100, standardize = False, **kwargs):
        if cv == "loo":
            n_splits = None
        else:
            n_splits = cv
        alpha_grid = np.logspace(-5, 5, 100)
        self.estimator = RidgeCV(alphas =alpha_grid, cv = n_splits, **kwargs)
        self.inv_link_fn = lambda a: a
        self.l_dot = lambda a, b: b - a
        self.l_doubledot = lambda a, b: 1
        self.r_doubledot = lambda a: 1
    
    def _fit_model(self, X, y):
        y_train = copy.deepcopy(y)
        sample_weight = np.ones_like(y_train)/(2 * len(y_train)) #for consistency with glmnet
        
        self.estimator.fit(X, y_train,sample_weight = sample_weight)
        self.coefficients_ = np.hstack([self.estimator.coef_, self.estimator.intercept_])
        self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        self.alpha_ = self.estimator.alpha_
        self.loo_coefficients_,self.influence_matrix_ = _get_loo_coefficients(X, y_train,orig_coef_ = self.coefficients_,alpha = self.alpha_,inv_link_fn = self.inv_link_fn,
                                                       l_doubledot = self.l_doubledot,r_doubledot = self.r_doubledot, l_dot = self.l_dot)
        
        

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pprint.pprint(f"X_train shape: {X_train.shape}")

    model = GlmNetElasticNetRegressorPPM()
    model.fit(X_train, y_train)

    #pprint.pprint(f"coefficients: {model.coefficients_}")

    preds = model.predict(X_test)

    pprint.pprint(f"r2 score: {r2_score(y_test, preds)}")
















            #Diag_ = 1.0/(evals + reg_curvature)
            #evecs_Diag = Diag_ * evecs
            #normal_eqn_mat = evecs_Diag @ X1_support.T

            #h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubled

    # def _get_aloocv_alpha(self, X, y,max_h = 1 - 1e-5):
    #     #Assume we are solving 1/n l_i + lambda * r
       
    #     all_support_idxs = {}
    #     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #     n = X1.shape[0]
    #     best_lambda_ = -1
    #     best_cv_ = np.inf
    #     best_loo_coefs = None
    #     best_influence_matrix = None
        
    #     #get min of self.estimator.lambda_path
    #     min_lambda_ = np.min(self.estimator.lambda_path_)

    #     for i,lambda_ in enumerate(self.estimator.lambda_path_):
    #         orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
    #         support_idxs_lambda_ = orig_coef_ != 0
    #         X1_support = X1[:, support_idxs_lambda_]
    #         if tuple(support_idxs_lambda_) not in all_support_idxs:
    #             evals, evecs = np.linalg.eigh(X1_support.T @ X1_support)
    #             all_support_idxs[tuple(support_idxs_lambda_)] = evals, evecs
    #         else:
    #             evals, evecs = all_support_idxs[tuple(support_idxs_lambda_)]
    #         orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
    #         l_doubledot_vals = self.l_doubledot(y, orig_preds)/n 
    #         orig_coef_ = orig_coef_[np.array(support_idxs_lambda_)]
    #         if self.r_doubledot is not None:
    #             r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
    #             #r_doubledot_vals[-1] = 0
    #             reg_curvature =  lambda_ * r_doubledot_vals
    #         else: 
    #             r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_) 
    #             #r_doubledot_vals[-1] = 0
    #             reg_curvature =  min_lambda_ * r_doubledot_vals

    #         Diag_ = 1.0/(evals + reg_curvature)
    #         evecs_Diag = Diag_ * evecs
    #         normal_eqn_mat = evecs_Diag @ X1_support.T

    #         h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubledot_vals
    #         h_vals[h_vals == 1] = max_h
    #         l_dot_vals = self.l_dot(y, orig_preds) / n
    #         influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)
    #         loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix 
    #         if not all(support_idxs_lambda_):
    #             loo_coef_dense_ = np.zeros((X1.shape[1], X.shape[0]))
    #             loo_coef_dense_[support_idxs_lambda_, :] = loo_coef_
    #             loo_coef_ = loo_coef_dense_
    #         sample_preds =  self.inv_link_fn(np.sum(loo_coef_.T * X1, axis=1))
    #         sample_scores = self.hyperparameter_scorer(y, sample_preds)
    #         if sample_scores < best_cv_:
    #             best_cv_ = sample_scores
    #             best_lambda_ = lambda_
    #             best_loo_coefs = loo_coef_
    #             best_influence_matrix = influence_matrix
        
    #     self.alpha_ = best_lambda_
    #     self.loo_coefficients_ = best_loo_coefs
    #     self.influence_matrix_ = best_influence_matrix
    





























    # def _fit_loo_coefficients(self, X, y, lambda_, max_h=1-1e-4,save_influence=False):
    #     """
    #     Get the coefficient (and intercept) for each LOO model. Since we fit
    #     one model for each sample, this gives an ndarray of shape (n_samples,
    #     n_features + 1)
    #     """
    #     orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
    #     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #     orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
    #     support_idxs = orig_coef_ != 0
    #     if not any(support_idxs):
    #         return orig_coef_ * np.ones_like(X1)
    #     X1 = X1[:, support_idxs]
    #     orig_coef_ = orig_coef_[support_idxs]
    #     l_doubledot_vals = self.l_doubledot(y, orig_preds)
    #     J = X1.T * l_doubledot_vals @ X1
    #     if self.r_doubledot is not None:
    #         r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
    #         r_doubledot_vals[-1] = 0
    #         reg_curvature = np.diag(r_doubledot_vals)
    #         J += lambda_ * reg_curvature
    #     inverse_J = np.linalg.inv(J) 
    #     normal_eqn_mat = inverse_J @ X1.T
    #     self.normal_eqn_mat = normal_eqn_mat
    #     h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
    #     h_vals[h_vals == 1] = max_h
    #     influence_matrix = normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
    #     loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix   #normal_eqn_mat * self.l_dot(y, orig_preds) / (1 - h_vals)
    #     if not all(support_idxs):
    #         loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
    #         loo_coef_dense_[support_idxs, :] = loo_coef_
    #         loo_coef_ = loo_coef_dense_
    #     if save_influence:
    #         return loo_coef_.T, influence_matrix
    #     else:
    #         return loo_coef_.T

    # #fit the model on the training set and compute the coefficients
        # if self.cv == "loo":
        #     self.loo_coefficients_, influence_matrix_ = self._fit_loo_coefficients(X, y_train, self.alpha_,save_influence=True)
        #     self.influence_matrix_ = influence_matrix_
        #     self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
        #     self.support_idxs_ =np.where(self.coefficients_ != 0)[0]
        # else:
        #     self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]



    # def _get_aloocv_alpha(self, X, y):
    #     cv_scores = np.zeros(len(self.estimator.lambda_path_))
    #     cv_scores_dict = {}
    #     loo_coefs = {}
    #     for i, lambda_ in tqdm(enumerate(self.estimator.lambda_path_),disable=True):
    #         loo_coef_ = self._fit_loo_coefficients(X, y, lambda_)
    #         loo_coefs[lambda_] = loo_coef_
    #         X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #         sample_scores = np.sum(loo_coef_ * X1, axis=1)
    #         preds = self.inv_link_fn(sample_scores)
    #         cv_scores[i] = self.hyperparameter_scorer(y, preds)
    #         cv_scores_dict[lambda_] =  self.hyperparameter_scorer(y, preds)


    # model = GlmNetRidgeRegressorPPM()
    # model.fit(X_train, y_train)

    # pprint.pprint(f"ridge coefficients: {model.coefficients_}")

    # preds = model.predict(X_test)

    # pprint.pprint(f"ridge r2 score: {r2_score(y_test, preds)}")

    # model = GlmNetLassoRegressorPPM()
    # model.fit(X_train, y_train)

    # pprint.pprint(f"lasso coefficients: {model.coefficients_}")

    # preds = model.predict(X_test)

    # pprint.pprint(f"laso r2 score: {r2_score(y_test, preds)}")


                #Sigma2_plus_lambda = s_squared + reg_curvature # p by p matrix 
                #Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda) # p by p matrix
                #normal_eqn_mat = v @ np.diag(Sigma2_plus_lambda_inverse) #p by p matrix 
                
                # if len(s) == X1_support.shape[1]: # low dimensional case
                #     l_double_dot_vals = np.zeros(X1_support.shape[1])
                #     l_double_dot_vals[:] = self.l_doubledot(y, orig_preds) 
                #     Sigma2_plus_lambda = s_squared + reg_curvature
                #     Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda)
                #     normal_eqn_mat = v @ np.diag(Sigma2_plus_lambda_inverse) @ sut # p by n
                # else: # high dimensional case
                #     l_double_dot_padded = np.zeros(X1_support.shape[1])
                #     l_double_dot_padded[:len(l_doubledot_vals)] = l_doubledot_vals
                #     Sigma2_plus_lambda = s_squared*l_double_dot_padded + reg_curvature
                #     normal_eqn_mat = normal_eqn_mat[:,X1.shape[0]] @ sut # p by n
                
                # s_inverse_lambda = 1.0/(np.diag(s_squared) + reg_curvature)
                # s_inverse_lambda_mat = np.zeros((X1_support.shape[1], X1_support.shape[1]))
                # p = X1_support.shape[1]
                # s_inverse_lambda_mat[:len(s_inverse_lambda), :len(s_inverse_lambda)] = np.diag(s_inverse_lambda)


                #normal_eqn_mat = np.dot(v, np.dot(np.diag(s_inverse_lambda), sut))
                
                #normal_eqn_mat = s_inverse_lambda_mat @ sut
                #normal_eqn_mat = v @ normal_eqn_mat
    

            #u,s,vh =  linalg.svd(X1_support,full_matrices=True)  #np.linalg.svd(X1_support,full_matrices = True)  #randomized_svd(X1_support, n_components= max(X1.shape[0], X1.shape[1])) 
            

            # s_mat = np.zeros((X1_support.shape[0], X1_support.shape[1]))
            # s_mat[:len(s), :len(s)] = s
            # s_squared = s_mat.T @ s_mat
            # sut = s_mat.T @ u.T

            # if len(s) == X1_support.shape[1]: # low dimensional case
            #     s_squared = np.diag(s**2) # p by p
            #     sut = np.diag(s) @ u.T[:len(s),:] # p by n matrix
            
            # else: # high dimensional case
            #     s_padded = np.zeros(X1_support.shape[1])
            #     s_padded[:len(s)] = s
            #     s_squared = np.diag(s_padded**2)
            #     sut = np.diag(s) @ u.T[:len(s),:] # n by n matrix. Should be a p by n matrix, but assume last p - n rows are 0

            #v = vh.T # p by p matrix

    
    # def _get_aloocv_alpha_test(self,X, y, max_h=1-1e-4, r = "all"):
    #     all_support_idxs = {}
        
    #     for lambda_ in self.estimator.lambda_path_:
    #         orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
    #         support_idxs_lambda_ = orig_coef_ != 0
    #         support_idxs_lambda_ = tuple(support_idxs_lambda_)
    #         if support_idxs_lambda_ not in all_support_idxs:
    #             all_support_idxs[support_idxs_lambda_] = []
    #         all_support_idxs[support_idxs_lambda_].append(lambda_)  

    #     #all_leave_out_coefs = np.zeros((len(self.estimator.lambda_path_),X.shape[0],X.shape[1] + 1))
    #     #all_leave_out_coefs_dict = {}
       
    #     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #     idx_to_lambda = OrderedDict()
       
        
    #     for support_idxs in all_support_idxs:
    #         lambda_path_support = all_support_idxs[support_idxs]
            
    #         X1_support = X1[:, support_idxs]
    #         stime = time.time()
    #         evals, evecs = np.linalg.eigh(X1_support.T @ X1_support)
            
    #         reg_curvatures = []
    #         all_orig_preds = []
    #         all_orig_coefs = np.zeros((len(lambda_path_support), X1_support.shape[1]))
    #         idx_to_lambda = OrderedDict()
    #         lambda_to_idx = OrderedDict()

    #         for i,lambda_ in enumerate(lambda_path_support):
    #             orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
                
    #             orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
    #             if not any(support_idxs):
    #                 return orig_coef_ * np.ones_like(X1)
    #             l_doubledot_vals = self.l_doubledot(y, orig_preds) 
                
                
    #             orig_coef_ = orig_coef_[np.array(support_idxs)]

    #             if self.r_doubledot is not None:
    #                 r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
    #                 r_doubledot_vals[-1] = 0
    #                 reg_curvature = lambda_ * r_doubledot_vals #np.diag(r_doubledot_vals)
    #             else:
    #                 reg_curvature = np.zeros(orig_coef_)
    #             reg_curvatures.append(reg_curvature)
    #             all_orig_preds.append(orig_preds)
    #             all_orig_coefs[i,:] = orig_coef_
    #             idx_to_lambda[i] = lambda_
    #             lambda_to_idx[lambda_] = i
            
    #         reg_curvatures_stacked = np.stack(reg_curvatures, axis=0)
    #         Diag_ = 1.0/(evals + reg_curvatures_stacked)
    #         #evecs_Diag = Diag_[:, :, np.newaxis] * evecs
    #         #print(evecs_Diag.shape)
    #         evecs_Diag = Diag_ @ evecs 
    #         print(evecs_Diag.shape)
    
    #         #normal_eqn_mat = np.einsum('ijk,kl->ijl', evecs_Diag, X1_support.T,optimize=True)
    #         normal_eqn_mat_dicts = {lambda_: evecs_Diag[i,:] * X1_support.T for i,lambda_ in enumerate(lambda_path_support)}


    #         print(lambda_to_idx[self.alpha_])
    #         normal_eqn_mat_old_alpha = normal_eqn_mat[lambda_to_idx[self.alpha_],:,:] - self.normal_eqn_mat
    #         print(normal_eqn_mat_old_alpha)

    #         print(normal_eqn_mat.shape)
    #         h_vals = np.sum(X1_support.T[np.newaxis, :, :] * normal_eqn_mat, axis=1)
            
    #         l_dot_vals = np.array([self.l_dot(y, preds) for preds in all_orig_preds])

           

    #         influence_matrix = normal_eqn_mat * l_dot_vals[:, np.newaxis, :] / (1 - h_vals[:, np.newaxis, :])

    #         all_loo_coef_ = all_orig_coefs[ :, :,np.newaxis] + influence_matrix
    #         all_loo_coef_ = all_loo_coef_.transpose(0,2,1)
    #         sample_preds = all_loo_coef_ * np.expand_dims(X1_support, 0) 
    #         sample_preds = np.sum(sample_preds, axis=2)
    #         sample_preds = self.inv_link_fn(sample_preds)
    #         cv_scores = [self.hyperparameter_scorer(y_true=y, y_pred= sample_preds[i,:]) for i in range(sample_preds.shape[0])]
    #         best_lambda_idx, best_cv_score = np.argmin(cv_scores), np.min(cv_scores)
    #         best_lambda = idx_to_lambda[best_lambda_idx]
    #         print(f"Best lambda is {best_lambda} with cv score {best_cv_score}")


            # for lambda_ in lambda_path_support:
                
            #     orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
                
            #     orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
            #     if not any(support_idxs):
            #         return orig_coef_ * np.ones_like(X1)
            #     l_doubledot_vals = self.l_doubledot(y, orig_preds) 
                
                
            #     orig_coef_ = orig_coef_[np.array(support_idxs)]

            #     if self.r_doubledot is not None:
            #         r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
            #         r_doubledot_vals[-1] = 0
            #         reg_curvature = lambda_ * np.diag(r_doubledot_vals)
            #     else:
            #         reg_curvature = np.zeros(orig_coef_) #1e-4 * np.ones_like(orig_coef_)
                
                
                
            #     Diag_ = np.diag(1.0/(evals + reg_curvature))
            #     s_time = time.time()
            #     normal_eqn_mat = np.dot(evecs * Diag_, evecs.T) @ X1_support.T
            #     total_normal_mat_time += time.time() - s_time
                
            #     s_time = time.time()
            #     h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubledot_vals
            #     total_h_val_time += time.time() - s_time
            #     h_vals[h_vals == 1] = max_h
            #     l_dot_vals = self.l_dot(y, orig_preds) 
            #     influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)
            #     loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix 
            #     if not all(support_idxs):
            #         loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
            #         loo_coef_dense_[support_idxs, :] = loo_coef_
            #         loo_coef_ = loo_coef_dense_
            #     all_leave_out_coefs[idx,:,:] = loo_coef_.T
            #     idx_to_lambda[idx] = lambda_
            #     #all_leave_out_coefs_dict[lambda_] = loo_coef_.T
            #     idx += 1

        #print(f"Total time taken for normal matrix is {total_normal_mat_time}")
        #print(f"Total time taken for h_vals is {total_h_val_time}")

        # sample_preds = all_leave_out_coefs * np.expand_dims(X1, 0) 
        # sample_preds = np.sum(sample_preds, axis=2)
        # sample_preds = self.inv_link_fn(sample_preds)
        # cv_scores = [self.hyperparameter_scorer(y_true=y, y_pred= sample_preds[i,:]) for i in range(sample_preds.shape[0])]
        # argmin_lambda = idx_to_lambda[np.argmin(cv_scores)]
        # print(f"Argmin lambda is {argmin_lambda}")
        # return argmin_lambda   

        # j = 0
        # cv_scores = np.zeros(len(self.estimator.lambda_path_))
        # cv_scores_dict = OrderedDict()
        # for lambda_ in all_leave_out_coefs_dict:
        #    loo_coef_ =  all_leave_out_coefs_dict[lambda_]
        #    sample_scores = np.sum(loo_coef_ * X1, axis=1)
        #    preds = self.inv_link_fn(sample_scores)
        #    cv_scores[j] = self.hyperparameter_scorer(y, preds)
        #    cv_scores_dict[j] = lambda_
        #    j+=1
        # return cv_scores_dict[np.argmin(cv_scores)]
        

        