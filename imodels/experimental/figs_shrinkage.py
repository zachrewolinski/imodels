from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from imodels.util import checks


class HSFIGS:
    def __init__(self, estimator_: BaseEstimator, reg_param: float = 1, shrinkage_scheme_: str = 'node_based'):
        """HSTree (Tree with hierarchical shrinkage applied).
        Hierarchical shinkage is an extremely fast post-hoc regularization method which works on any decision tree (or tree-based ensemble, such as Random Forest).
        It does not modify the tree structure, and instead regularizes the tree by shrinking the prediction over each node towards the sample means of its ancestors (using a single regularization parameter).
        Experiments over a wide variety of datasets show that hierarchical shrinkage substantially increases the predictive performance of individual decision trees and decision-tree ensembles.
        https://arxiv.org/abs/2202.00858

        Params
        ------
        estimator_: sklearn tree or tree ensemble model (e.g. RandomForest or GradientBoosting)

        reg_param: float
            Higher is more regularization (can be arbitrarily large, should not be < 0)

        shrinkage_scheme: str
            Experimental: Used to experiment with different forms of shrinkage. options are:
                (i) node_based shrinks based on number of samples in parent node
                (ii) leaf_based only shrinks leaf nodes based on number of leaf samples
                (iii) constant shrinks every node by a constant lambda
        """
        super().__init__()
        self.reg_param = reg_param
        # print('est', estimator_)
        self.estimator_ = estimator_
        self.shrinkage_scheme_ = shrinkage_scheme_
        self._init_prediction_task()

        if checks.check_is_fitted(self.estimator_):
            self._shrink()

    def __init__prediction_task(self):
        self.prediction_task = 'regression'

    def get_params(self, deep=True):
        if deep:
            return deepcopy({'reg_param': self.reg_param, 'estimator_': self.estimator_,
                             # 'prediction_task': self.prediction_task,
                             'shrinkage_scheme_': self.shrinkage_scheme_})
        return {'reg_param': self.reg_param, 'estimator_': self.estimator_,
                # 'prediction_task': self.prediction_task,
                'shrinkage_scheme_': self.shrinkage_scheme_}

    def fit(self, *args, **kwargs):
        self.estimator_.fit(*args, **kwargs)
        self._shrink()

    def _shrink(self, reg_param):
        for tree in self.trees_:
            tree.shrink(reg_param)

    def predict(self, *args, **kwargs):
        return self.estimator_.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented


class HSFIGSRegressor(HSFIGS):
    def _init_prediction_task(self):
        self.prediction_task = 'regression'


class HSFIGSClassifier(HSFIGS):
    def _init_prediction_task(self):
        self.prediction_task = 'classification'


class HSFIGSClassifierCV(HSFIGSClassifier):
    def __init__(self, estimator_: BaseEstimator,
                 reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500], shrinkage_scheme_: str = 'node_based',
                 cv: int = 3, scoring=None, *args, **kwargs):
        """Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.
        """
        super().__init__(estimator_, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y, *args, **kwargs):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSFIGSClassifier(deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)


class HSFIGSRegressorCV(HSFIGSRegressor):
    def __init__(self, estimator_: BaseEstimator,
                 reg_param_list: List[float] = [0.1, 1, 10, 50, 100, 500],
                 shrinkage_scheme_: str = 'node_based',
                 cv: int = 3, scoring=None, *args, **kwargs):
        """Note: args, kwargs are not used but left so that imodels-experiments can still pass redundant args.
        Cross-validation is used to select the best regularization parameter for hierarchical shrinkage.
        """
        super().__init__(estimator_, reg_param=None)
        self.reg_param_list = np.array(reg_param_list)
        self.cv = cv
        self.scoring = scoring
        self.shrinkage_scheme_ = shrinkage_scheme_
        # print('estimator', self.estimator_,
        #       'checks.check_is_fitted(estimator)', checks.check_is_fitted(self.estimator_))
        # if checks.check_is_fitted(self.estimator_):
        #     raise Warning('Passed an already fitted estimator,'
        #                   'but shrinking not applied until fit method is called.')

    def fit(self, X, y):
        self.scores_ = []
        for reg_param in self.reg_param_list:
            est = HSFIGSRegressor(deepcopy(self.estimator_), reg_param)
            cv_scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            self.scores_.append(np.mean(cv_scores))
        self.reg_param = self.reg_param_list[np.argmax(self.scores_)]
        super().fit(X=X, y=y)
