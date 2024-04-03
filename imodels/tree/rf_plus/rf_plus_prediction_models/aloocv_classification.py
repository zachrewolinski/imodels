# Generic Imports 
import copy, pprint, warnings, imodels
from abc import ABC, abstractmethod
from functools import partial
import time, numbers
import numpy as np
import scipy as sp
import pandas as pd
from collections import OrderedDict


#scipy imports
from scipy.special import softmax, expit
from scipy import linalg

# Sklearn Imports
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, f1_score, accuracy_score

#Glmnet Imports
from glmnet import LogitNet

#imodels imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.SVM_wrapper import CustomSVMClassifier, derivative_squared_hinge_loss, second_derivative_squared_hinge_loss



class AloGLMClassifier(AloGLM):
    """
    Only can deal with binary for now. 
    """
    
    def __init__(self, estimator, cv="loo", n_alphas=100, l1_ratio=0.0, standardize = False,
                 inv_link_fn=lambda a: a, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: 1, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=log_loss, class_weight = 'balanced'):
        
        #estimator.__init__()
        
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
        self.class_weight = class_weight

    def fit(self, X, y):

        y_train = copy.deepcopy(y)
        
        #Process Sample Weights
        if self.class_weight is None:
            self.sample_weights = np.ones(len(y_train))
        else:
            if self.class_weight == 'balanced':
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
            self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y,self.coefficients_,self.alpha_)
            self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        else:
            self.alpha_ = self.estimator.lambda_best_[0]
            self.coefficients_ = self._augmented_coeffs_for_each_alpha[self.alpha_]
            self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
       
    def predict_proba(self, X):
        preds = self.predict(X)
        return np.stack([1 - preds, preds]).T
        #return preds

    
class AloLogisticElasticNetClassifier(AloGLMClassifier):
    """
    PPM class for Logistic Regression with Elastic Net Penalty
    """

    def __init__(self, estimator = LogitNet, cv="loo", n_alphas=100, l1_ratio=0.5, standardize=False, inv_link_fn= expit, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: b * (1-b), r_doubledot=lambda a: 1, hyperparameter_scorer=log_loss, class_weight='balanced'):
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = l1_ratio,standardize = standardize), 
                         cv, n_alphas, l1_ratio, standardize, inv_link_fn, l_dot, l_doubledot, r_doubledot, 
                         hyperparameter_scorer, class_weight)
        
       
class AloLogisticRidgeClassifier(AloGLMClassifier):
    """
    PPM class for Logistic Regression with Ridge Penalty
    """

    def __init__(self, estimator = LogitNet, cv="loo", n_alphas=100, l1_ratio=0, standardize=True, inv_link_fn=expit, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: b * (1-b), r_doubledot=lambda a: 1, hyperparameter_scorer=log_loss, class_weight='balanced'):
        
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv

        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 0,standardize = standardize), 
                         cv, n_alphas, l1_ratio, standardize, inv_link_fn, l_dot, l_doubledot, r_doubledot, 
                         hyperparameter_scorer, class_weight)


class AloLogisticLassoClassifier(AloGLMClassifier):
    """
    PPM class for Logistic Regression with Lasso Penalty
    """


    def __init__(self, estimator = LogitNet, cv="loo", n_alphas=100, l1_ratio=1.0, standardize=False, inv_link_fn=expit, l_dot=lambda a, b: b - a, 
                 l_doubledot=lambda a, b: b * (1-b), r_doubledot=lambda a: None, hyperparameter_scorer=log_loss, class_weight='balanced'):
        
        if cv == "loo":
            n_splits = 0
        else:
            n_splits = cv
        
        super().__init__(LogitNet(n_lambda=n_alphas, n_splits=n_splits, alpha = 1,standardize = standardize), 
                         cv, n_alphas, l1_ratio, standardize, inv_link_fn, l_dot, l_doubledot, r_doubledot, 
                         hyperparameter_scorer, class_weight)


class AloSVCRidgeClassifier(AloGLMClassifier):
    """
    SVM Classifier
    """

    def __init__(self, estimator = CustomSVMClassifier, cv="loo", n_alphas=20, standardize=False,
                  inv_link_fn=expit, l_dot=derivative_squared_hinge_loss, dual="auto",
                 l_doubledot=second_derivative_squared_hinge_loss, r_doubledot=lambda a: 1,
                 hyperparameter_scorer=log_loss, class_weight='balanced'):
        
        super().__init__(CustomSVMClassifier(n_alphas=n_alphas,dual = dual), cv, n_alphas, 0, standardize,
                        inv_link_fn, l_dot, l_doubledot, r_doubledot, hyperparameter_scorer, class_weight)
        
   
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    

    model = AloLogisticRidgeClassifier()
    model.fit(X_train, y_train)
    model.predict_proba(X_test)

