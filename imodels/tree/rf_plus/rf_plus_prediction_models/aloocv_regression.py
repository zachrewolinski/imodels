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
from scipy.special import softmax
from scipy import linalg

# Sklearn Imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.linear_model import RidgeCV, HuberRegressor

#Glmnet Imports
from glmnet import ElasticNet

#imodels imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.SVM_wrapper import CustomSVMClassifier, derivative_squared_hinge_loss, second_derivative_squared_hinge_loss


class AloGLMRegressor(AloGLM,ABC):
    """
    Alo Regressor
    """
    pass
     



class AloElasticNetRegressor(AloGLMRegressor):
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
        
class AloGLMRidgeRegressor(AloGLMRegressor):
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


class AloLassoRegressor(AloGLMRegressor):
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
    

class AloRidgeRegressor(AloGLMRegressor):
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
    
    def fit(self, X, y):
        y_train = copy.deepcopy(y)
        sample_weight = np.ones_like(y_train)/(2 * len(y_train)) #for consistency with glmnet
        
        self.estimator.fit(X, y_train,sample_weight = sample_weight)
        self.coefficients_ = np.hstack([self.estimator.coef_, self.estimator.intercept_])
        self.support_idxs_ = np.where(self.coefficients_ != 0)[0]
        self.alpha_ = self.estimator.alpha_
        self.loo_coefficients_,self.influence_matrix_ = self._get_loo_coefficients(X, y_train,orig_coef_ = self.coefficients_,alpha = self.alpha_)
        


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
        

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pprint.pprint(f"X_train shape: {X_train.shape}")

    model = AloGLMRidgeRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    pprint.pprint(f"r2 score: {r2_score(y_test, preds)}")

