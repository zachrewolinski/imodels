import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import scipy as sp
import pandas as pd
import time, numbers
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, HuberRegressor, Lasso
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score, log_loss
from scipy.special import softmax
from scipy import sparse
from glmnet import LogitNet
warnings.filterwarnings("ignore")
from scipy.special import expit, logit
from scipy.linalg import lstsq as sp_lstsq
from sklearn.linear_model import Ridge

from imodels.tree.rf_plus.ppms.ppms import PartialPredictionModelBase, _GlmPPM

from imodels.tree.rf_plus.ppms.ppms_util import _extract_coef_and_intercept, _get_preds, fast_hessian_vector_inverse, count_sketch_inverse, _get_loo_coefficients
from scipy.sparse.linalg import spsolve,cg
from scipy.linalg import solve
from sklearn import random_projection
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from collections import OrderedDict
from scipy import linalg
from imodels.tree.rf_plus.ppms.SVM_Classifier_Wrapper import CustomSVMClassifier, derivative_squared_hinge_loss, second_derivative_squared_hinge_loss

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
        self.influence_matrix_ = None
        self.support_idxs = None
        
    def _fit_model(self, X, y):

        y_train = y
        
        #Process Sample Weights
        if self.sample_weights is None:
            self.sample_weights = np.ones(len(y_train))
        else:
            if self.sample_weights == 'balanced':
                n_pos = np.sum(y_train)
                n_neg = len(y_train) - n_pos
                weight_pos = (1 / n_pos) * (n_pos + n_neg) / 2.0
                weight_neg = (1 / n_neg) * (n_pos + n_neg) / 2.0
                self.sample_weights = np.where(y_train == 1, weight_pos, weight_neg) * len(y_train)
            else:
                assert len(self.sample_weights) == len(y_train), "Sample weights must be the same length as the target"
        
        
        
        self.estimator.fit(X, y_train, sample_weight=self.sample_weights)   

        
        for i,lambda_ in enumerate(self.estimator.lambda_path_):
            self._augmented_coeffs_for_each_alpha[lambda_] = np.hstack([self.estimator.coef_path_[0,:, i],self.estimator.intercept_path_[0,i]]) #only binary classification        


        #fit the model on the training set and compute the coefficients
        if self.cv == "loo":
            self._get_aloocv_alpha(X, y)
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
            self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        else:
            self.alpha_ = self.estimator.lambda_best_[0]
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
            self.support_idxs_ = np.where(self.coefficients_ != 0)[0]

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
        

       
    def predict_proba(self, X):
        preds = self.predict(X)
        return preds
        #return np.stack([1 - preds, preds]).T

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
        if use_loo:
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

    def __init__(self, cv="loo", n_alphas=100, standardize = False, sample_weights = None,**kwargs): #'balanced'
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 0,standardize = standardize, **kwargs), 
                        cv,inv_link_fn=sp.special.expit, l_dot=lambda a, b: b - a, l1_ratio=0, sample_weights = sample_weights,
                        l_doubledot=lambda a, b:  b * (1-b), r_doubledot= lambda a: 1 ,
                        hyperparameter_scorer=log_loss)



class SVCRidgePPM(GlmClassifierPPM, CustomSVMClassifier):
    """
    PPM class for SVM Classifier
    """

    def __init__(self, cv="loo", n_alphas=20,sample_weights = 'balanced',**kwargs):
        if cv == "loo":
            n_splits = 0
        else:
            raise ValueError("Only leave one out cross validation is supported for SVM")
        
        super().__init__(CustomSVMClassifier(n_alphas=n_alphas, **kwargs), 
                        cv,inv_link_fn=sp.special.expit, l_dot= derivative_squared_hinge_loss,
                        l_doubledot= second_derivative_squared_hinge_loss, r_doubledot= lambda a: 1,
                        sample_weights = sample_weights,hyperparameter_scorer=log_loss)
        
   
if __name__ == "__main__":
    # from sklearn.datasets import make_regression
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    

    model = GLMLogisticElasticNetPPM(l1_ratio=0.5)
    model.fit(X_train, y_train)
    print(model.influence_matrix_.shape)

    model = SVCRidgePPM()
    model.fit(X_train, y_train)


# # Not sure if sample weights make sense during leave one out cross validation. 
#     def _get_aloocv_alpha(self, X, y,max_h = 1 - 1e-4):
        
#         all_support_idxs = {}
#         X1 = np.hstack([X, np.ones((X.shape[0], 1))])
#         cv_scores = []
#         best_lambda_ = -1
#         best_cv_ = np.inf
#         best_loo_coefs = None
#         best_influence_matrix = None
#         min_lambda_ = np.min(self.estimator.lambda_path_)

#         #self.sample_weights = np.ones(y.shape[0])

#         for i,lambda_ in enumerate(self.estimator.lambda_path_):
#             orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
#             support_idxs_lambda_ = orig_coef_ != 0
#             X1_support = X1[:, support_idxs_lambda_]
#             if tuple(support_idxs_lambda_) not in all_support_idxs:
#                 evals, evecs = np.linalg.eigh(X1_support.T @ X1_support)
#                 all_support_idxs[tuple(support_idxs_lambda_)] = evals, evecs
#             else:
#                 evals, evecs = all_support_idxs[tuple(support_idxs_lambda_)]
#             orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
#             l_doubledot_vals = self.l_doubledot(y, orig_preds) #* self.sample_weights
#             orig_coef_ = orig_coef_[np.array(support_idxs_lambda_)]
#             if self.r_doubledot is not None:
#                 r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
#                 r_doubledot_vals[-1] = 0
#                 reg_curvature = lambda_ * r_doubledot_vals
#             else:  
#                 reg_curvature = min_lambda_ * np.ones_like(orig_coef_)
#             Diag_ = 1.0/(evals + reg_curvature)
#             evecs_Diag = Diag_ * evecs
#             normal_eqn_mat = evecs_Diag @ X1_support.T

#             h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubledot_vals 
#             h_vals[h_vals == 1] = max_h
#             l_dot_vals = self.l_dot(y, orig_preds) #* self.sample_weights
#             influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)
#             loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix 
#             if not all(support_idxs_lambda_):
#                 loo_coef_dense_ = np.zeros((X1.shape[1], X.shape[0]))
#                 loo_coef_dense_[support_idxs_lambda_, :] = loo_coef_
#                 loo_coef_ = loo_coef_dense_
            
#             sample_preds =  self.inv_link_fn(np.sum(loo_coef_.T * X1, axis=1))
#             sample_scores = self.hyperparameter_scorer(y, sample_preds)#,sample_weight=self.sample_weights)
            
#             if sample_scores < best_cv_:
#                 best_cv_ = sample_scores
#                 best_lambda_ = lambda_
#                 best_loo_coefs = loo_coef_
#                 best_influence_matrix = influence_matrix
        
#         self.alpha_ = best_lambda_
#         self.loo_coefficients_ = best_loo_coefs
#         self.influence_matrix_ = best_influence_matrix

#         #self.alpha_ = best_lambda_
#         #self.loo_coefficients_ = best_loo_coefs
#         #self.influence_matrix_ = best_influence_matrix
    







































    # def _fit_loo_coefficients(self, X, y, lambda_, max_h=1-1e-4, save_influence=False):
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
    #     l_doubledot_vals = self.l_doubledot(y, orig_preds) * self.sample_weights

    #     J = X1.T * l_doubledot_vals @ X1
    #     if self.r_doubledot is not None:
    #         r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
    #         r_doubledot_vals[-1] = 0
    #         reg_curvature = np.diag(r_doubledot_vals)
    #         J += lambda_ * reg_curvature
        
    #     normal_eqn_mat = np.linalg.inv(J) @ X1.T
    #     h_vals = np.sum(X1.T * normal_eqn_mat, axis=0) * l_doubledot_vals
    #     h_vals[h_vals == 1] = max_h
    #     l_dot_vals = self.l_dot(y, orig_preds) 
    #     influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)
    #     loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix * self.sample_weights
    #     #l_dot_vals = self.l_dot(y, orig_preds) * self.sample_weights
    #     #loo_coef_ = orig_coef_[:, np.newaxis] + normal_eqn_mat * l_dot_vals / (1 - h_vals)
    #     if not all(support_idxs):
    #         loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
    #         loo_coef_dense_[support_idxs, :] = loo_coef_
    #         loo_coef_ = loo_coef_dense_
    #     if save_influence:
    #         return loo_coef_.T, influence_matrix
    #     else:
    #         return loo_coef_.T


    # def _get_aloocv_alpha(self, X, y):
    #     cv_scores = np.zeros(len(self.estimator.lambda_path_))
    #     for i, lambda_ in enumerate(self.estimator.lambda_path_):
    #         loo_coef_ = self._fit_loo_coefficients(X, y, lambda_)
    #         X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #         sample_scores = np.sum(loo_coef_ * X1, axis=1)
    #         preds = self.inv_link_fn(sample_scores)
    #         cv_scores[i] = self.hyperparameter_scorer(y, preds,sample_weight=self.sample_weights)
    #     return self.estimator.lambda_path_[np.argmin(cv_scores)]
    


    
    # def _get_aloocv_alpha_test(self,X, y, max_h=1-1e-4, r = "all"):
    #     all_support_idxs = {}
        
    #     for lambda_ in self.estimator.lambda_path_:
    #         orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
    #         support_idxs_lambda_ = orig_coef_ != 0
    #         support_idxs_lambda_ = tuple(support_idxs_lambda_)
    #         if support_idxs_lambda_ not in all_support_idxs:
    #             all_support_idxs[support_idxs_lambda_] = []
    #         all_support_idxs[support_idxs_lambda_].append(lambda_)  

    #     all_leave_out_coefs = np.zeros((len(self.estimator.lambda_path_),X.shape[0],X.shape[1] + 1))
    #     all_leave_out_coefs_dict = {}
       
    #     X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    #     idx_to_lambda = {}
    #     idx = 0
    #     for support_idxs in all_support_idxs:
    #         lambda_path_support = all_support_idxs[support_idxs]
            
    #         X1_support = X1[:, support_idxs]
            

    #         u,s,vh =  linalg.svd(X1_support,full_matrices=True)  #np.linalg.svd(X1_support,full_matrices = True)  #randomized_svd(X1_support, n_components= max(X1.shape[0], X1.shape[1])) 
            

    #         if len(s) == X1_support.shape[1]: # low dimensional case
    #             s_squared = np.diag(s**2) # p by p
    #             sut = np.diag(s) @ u.T[:len(s),:] # p by n matrix
            
    #         else: # high dimensional case
    #             s_padded = np.zeros(X1_support.shape[1])
    #             s_padded[:len(s)] = s
    #             s_squared = np.diag(s_padded**2)
    #             sut = np.diag(s) @ u.T[:len(s),:] # n by n matrix. Should be a p by n matrix, but assume last p - n rows are 0

    #         v = vh.T # p by p matrix


    #         for lambda_ in lambda_path_support:
                
    #             orig_coef_ = self._augmented_coeffs_for_each_alpha[lambda_]
                
    #             orig_preds = _get_preds(X, orig_coef_, self.inv_link_fn)
    #             if not any(support_idxs):
    #                 return orig_coef_ * np.ones_like(X1)
    #             l_doubledot_vals = self.l_doubledot(y, orig_preds) * self.sample_weights
                
                
    #             orig_coef_ = orig_coef_[np.array(support_idxs)]

    #             if self.r_doubledot is not None:
    #                 r_doubledot_vals = self.r_doubledot(orig_coef_) * np.ones_like(orig_coef_)
    #                 r_doubledot_vals[-1] = 0
    #                 reg_curvature = lambda_ * np.diag(r_doubledot_vals)
                
                
    #             #Sigma2_plus_lambda = s_squared + reg_curvature # p by p matrix 
    #             #Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda) # p by p matrix
    #             #normal_eqn_mat = v @ np.diag(Sigma2_plus_lambda_inverse) #p by p matrix 
                
    #             if len(s) == X1_support.shape[1]: # low dimensional case
    #                 Sigma2_plus_lambda = s_squared*l_doubledot_vals[:len(s)] + reg_curvature
    #                 Sigma2_plus_lambda_inverse = 1.0/(Sigma2_plus_lambda)
    #                 normal_eqn_mat = v @ np.diag(Sigma2_plus_lambda_inverse) @ sut # p by n
    #             else: # high dimensional case
    #                 l_double_dot_padded = np.zeros(X1_support.shape[1])
    #                 l_double_dot_padded[:len(l_doubledot_vals)] = l_doubledot_vals
    #                 Sigma2_plus_lambda = s_squared*l_double_dot_padded + reg_curvature
    #                 normal_eqn_mat = normal_eqn_mat[:,X1.shape[0]] @ sut # p by n
                            
    #             if lambda_ == self.alpha_:
    #                 print(normal_eqn_mat)
                
    #             #normal_eqn_mat = np.dot(v, np.dot(np.diag(s_inverse_lambda), sut))

    #             h_vals = np.sum(X1_support.T * normal_eqn_mat, axis=0) * l_doubledot_vals
    #             h_vals[h_vals == 1] = max_h
    #             l_dot_vals = self.l_dot(y, orig_preds)  * self.sample_weights
    #             influence_matrix = normal_eqn_mat * l_dot_vals / (1 - h_vals)
    #             loo_coef_ = orig_coef_[:, np.newaxis] + influence_matrix * self.sample_weights
    #             if not all(support_idxs):
    #                 loo_coef_dense_ = np.zeros((X.shape[1] + 1, X.shape[0]))
    #                 loo_coef_dense_[support_idxs, :] = loo_coef_
    #                 loo_coef_ = loo_coef_dense_
    #             all_leave_out_coefs[idx,:,:] = loo_coef_.T
    #             idx_to_lambda[idx] = lambda_
    #             idx += 1
        
    


    # model = GLMLogisticLassoNetPPM()
    # model.fit(X_train, y_train)

    # model = GLMLogisticRidgeNetPPM()
    # model.fit(X_train, y_train)
    # print(roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))    


#pprint.pprint(f"Starting Matrix Inversion")

        #inverse_prod = fast_hessian_vector_inverse(J, X1.T)
        #inverse_prod = solve(J, X1.T, assume_a='sym',check_finite=False)
        #sp_lstsq(X, y, lapack_driver='gelsy', check_finite=False) #np.linalg.lstsq(J, X1.T, rcond=None)[0]
        #time_taken = time.time() - start
        #pprint.pprint(f"numpy least squares Time taken is {time_taken}")

        # start = time.time()
        # inverse_prod =  spsolve(J, X1.T, check_finite=False)
        # time_taken = time.time() - start
        # pprint.pprint(f"Alternative taken is {time_taken}")
        # print(J.shape)
    
#  start = time.time()
#         inverse_prod = Ridge(alpha=lambda_, fit_intercept=False,copy_X=False).fit(J, X1.T).coef_
#         time_taken = time.time() - start
#         pprint.pprint(f"CG Time taken is {time_taken}")
#         start = time.time()
    

    #s_padded = np.zeros(X1_support.shape[1])
            #s_padded[:len(s)] = s**2

            #S_mat = np.zeros((X1_support.shape[0], r))
            #S_mat[:len(s), :len(s)] = s
            #S_mat_squared = S_mat.T @ S_mat
    



            # if len(s) == v.shape[0]: # low dimensional case, len(s) = X1_support.shape[1] < X1_support.shape[0]
            #     s_squared = np.diag(s**2)
            #     sut =  np.diag(s) @ u.T
                
            # else: # high dimensional case, len(s) = X1_support.shape[0] < X1_support.shape[1]
            #     s_squared = np.diag(np.pad(s**2, (0, v.shape[0] - s.shape[0]), 'constant'))
            #     sut =  np.diag(s) @ u.T
           
            
            # if  min(X1_support.shape[0],X1_support.shape[1]) == X1_support.shape[0]: 
            #     s_n = s
            #     s_p = np.zeros(X1_support.shape[1])
            #     s_p[:len(s)] = s
            # else: # low dimensional case, len(s) = X1_support.shape[1]
            #     s_p = s
            #     s_n = np.zeros(X1_support.shape[0])
            #     s_n[:len(s)] = s

            # sut = np.diag(s_n) @ u.T 