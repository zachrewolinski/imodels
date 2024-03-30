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

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


class SGDClassifierPPM(SGDClassifier):
    pass
    # def __init__(self, loss="hinge", penalty="elasticnet", n_alphas = 5, l1_ratio=0.15,
    #              fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
    #              verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
    #              learning_rate="optimal", eta0=0.0, power_t=0.5, early_stopping=False,
    #              validation_fraction=0.1, n_iter_no_change=5, class_weight="balanced"):
        
    
    #     self.loss = loss
    #     self.penalty = penalty
    #     self.l1_ratio = l1_ratio
    #     self.fit_intercept = fit_intercept
    #     self.max_iter = max_iter
    #     self.tol = tol
    #     self.shuffle = shuffle
    #     self.verbose = verbose
    #     self.epsilon = epsilon
    #     self.n_jobs = n_jobs
    #     self.random_state = random_state
    #     self.learning_rate = learning_rate
    #     self.eta0 = eta0
    #     self.power_t = power_t
    #     self.early_stopping = early_stopping
    #     self.validation_fraction = validation_fraction
    #     self.n_iter_no_change = n_iter_no_change
    #     self.class_weight = class_weight
    #     self.n_alphas = n_alphas
    #     self.lambda_path_ = np.logspace(-3, 3, self.n_alphas)
    #     self.estimators_ = []
    #     for alpha in self.lambda_path_:
    #         self.estimators_.append(SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
    #                                               fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle,
    #                                               verbose=verbose, epsilon=epsilon, n_jobs=n_jobs, random_state=random_state,
    #                                               learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping,
    #                                               validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight))
    #     self.coef_path_ = None
    #     self.intercept_path_ = None
    
    # def fit(self, X, y):
    #     for estimator in self.estimators_:
    #         estimator.fit(X, y)
    #     return self
