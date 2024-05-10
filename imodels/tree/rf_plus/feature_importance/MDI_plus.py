# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import pdist
# from functools import partial

# # Sklearn imports
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.model_selection import train_test_split

# #imports from imodels
# import imodels
# from imodels.tree.rf_plus.feature_importance.MDI_plus import MDIPlusGenericRegressorPPM, MDIPlusGenericClassifierPPM, MDIPlusAloPartialPredictionModelRegressor, MDIPlusAloPartialPredictionModelClassifier
# from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloGLMRegressor, AloElasticNetRegressorCV
# from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV
# from imodels.tree.rf_plus.data_transformations.block_transformers import BlockTransformerBase, _blocked_train_test_split, BlockPartitionedData
# from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
# from imodels.tree.rf_plus.rf_plus.rf_plus_utils import _get_sample_split_data











# class DecisionTreeMDIPlus:
#     """
#     The class object for computing MDI+ feature importances for a single tree.
#     Generalized mean decrease in impurity (MDI+) is a flexible framework for computing RF
#     feature importances. For more details, refer to [paper].

#     Parameters
#     ----------
#     estimator: a fitted PartialPredictionModelBase object or scikit-learn type estimator
#         The fitted partial prediction model to use for evaluating
#         feature importance via MDI+. If not a PartialPredictionModelBase, then
#         the estimator is coerced into a PartialPredictionModelBase object via
#         GenericRegressorPPM or GenericClassifierPPM depending on the specified
#         task. Note that these generic PPMs may be computationally expensive.
#     transformer: a BlockTransformerBase object
#         A block feature transformer used to generate blocks of engineered
#         features for each original feature. The transformed data is then used
#         as input into the partial prediction models.
#     scoring_fns: a function or dict with functions as value and function name (str) as key
#         The scoring functions used for evaluating the partial predictions.
#     local_scoring_fns: one of True, False, function or dict with functions as value and function name (str) as key.
#         The local scoring functions used for evaluating the partial predictions per sample.
#         If False, then local feature importances are not evaluated.
#         If True, then the (global) scoring functions are used as the local scoring functions.
#         If a function is provided, this function is used as the local scoring function and applied per sample.
#         Otherwise, a dictionary of local scoring functions can be supplied, with one local scoring function per
#         global scoring function (using the same key).
#     sample_split: string in {"loo", "oob", "inbag"} or None
#         The sample splitting strategy to be used when evaluating the partial
#         model predictions. The default "loo" (leave-one-out) is strongly
#         recommended for performance and in particular, for overcoming the known
#         correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
#         also be used to overcome these biases. "inbag" is the sample splitting
#         strategy used by MDI. If None, no sample splitting is performed and the
#         full data set is used to evaluate the partial model predictions.
#     tree_random_state: int or None
#         Random state of the fitted tree; used in sample splitting and
#         only required if sample_split = "oob" or "inbag".
#     mode: string in {"keep_k", "keep_rest"}
#         Mode for the method. "keep_k" imputes the mean of each feature not
#         in block k when making a partial model prediction, while "keep_rest"
#         imputes the mean of each feature in block k. "keep_k" is strongly
#         recommended for computational considerations.
#     task: string in {"regression", "classification"}
#         The supervised learning task for the RF model. Used for choosing
#         defaults for the scoring_fns. Currently only regression and
#         classification are supported.
#     center: bool
#         Flag for whether to center the transformed data in the transformers.
#     normalize: bool
#         Flag for whether to rescale the transformed data to have unit
#         variance in the transformers.
#     """

#     def __init__(self, transformer, scoring_fns,
#                   tree_random_state=None, mode="keep_k", 
#                  task="regression", center=True, normalize=False):
       
#         assert mode in ["keep_k", "keep_rest"]
#         assert task in ["regression", "classification"]

#         self.transformer = transformer
#         self.scoring_fns = scoring_fns
#         self.tree_random_state = tree_random_state
#         self.mode = mode
#         self.task = task
#         self.center = center
#         self.normalize = normalize
#         self.n_features = None
    
#     def fit(self, X, y, estimator, use_loo = True):
#         """
#         Fit the MDI+ feature importance object.

#         X,y should be the same data that was used to train the tree! 

#         If use_loo is True, then we use LOO to compute the feature importances. 
#         The linear estimator should have LOO coefficients. 

#         Parameters
#         ----------
#         X: array-like of shape (n_samples, n_features)
#             The input data to be used for fitting the MDI+ object.
#         y: array-like of shape (n_samples,)
#             The target values to be used for fitting the MDI+ object.
#         """
#         #Initialize partial predictive model and check if it is an AloPPM
#         ppm, is_aloppm = self._initialize_ppm(estimator)
        

#         # Transform data
#         blocked_data = self.transformer.transform(X, center=self.center,
#                                                   normalize=self.normalize)
#         self.n_features = blocked_data.n_blocks
#         in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, in_bag_indices,oob_indices = _get_sample_split_data(blocked_data, y, self.tree_random_state)
        
        
#         if in_bag_blocked_data.shape[1] == 0:
#             raise ValueError("No features were generated by the transformer. Please check the transformer parameters.")
         
#         # Get the partial predictions
#         if use_loo and is_aloppm:
#             partial_preds = ppm.predict_partial(blocked_data,self.mode,use_loo=use_loo)
#         else:
#             partial_preds = ppm.predict_partial(in_bag_blocked_data, mode=self.mode)



    
    
    
#     def _initialize_ppm(self, estimator):
#         is_aloppm = False
#         if self.task == "regression":
#             if isinstance(estimator, AloGLMRegressor):
#                 ppm = AloGLMRegressor(estimator)
#                 is_aloppm = True
#             else:
#                 ppm = MDIPlusGenericRegressorPPM(estimator)
#         else:
#             if isinstance(estimator, AloGLMClassifier):
#                 ppm = AloGLMClassifier(estimator)
#                 is_aloppm = True
#             else:
#                 ppm = MDIPlusGenericClassifierPPM(estimator)
#         return ppm, is_aloppm




#         # Fit the partial prediction model
#         # if use_loo and hasattr(self.estimator, "loo_coefficients_"):
#         #     pass
#         # else:
#         #     if self.task == "regression":
#         #         ppm = GenericRegressorPPM(estimator) 
#         #     else:
#         #         ppm = GenericClassifierPPM(estimator)

#         #     partial_preds = self.estimator.predict_partial(train_blocked_data, mode=self.mode)  


# def _partial_preds_to_scores(partial_preds, y_test, scoring_fn):
#     scores = []
#     for k, y_pred in partial_preds.items():
#         if isinstance(y_pred, tuple):  # if constant model
#             y_pred = np.ones_like(y_test) * y_pred[1]
#         scores.append(scoring_fn(y_test, y_pred))
#     return np.vstack(scores)     



















#  #     if hasattr(self.estimator, "predict_full") and hasattr(self.estimator, "predict_partial"):
#         #         full_preds = self.estimator.predict_full(test_blocked_data)
#         #         partial_preds = self.estimator.predict_partial(test_blocked_data, mode=self.mode)
#         #     else:
#         #         if self.task == "regression":
#         #             ppm = GenericRegressorPPM(self.estimator)
#         #         elif self.task == "classification":
#         #             ppm = GenericClassifierPPM(self.estimator)
#         #         full_preds = ppm.predict_full(test_blocked_data)
#         #         partial_preds = ppm.predict_partial(test_blocked_data, mode=self.mode)
#         #     self._score_full_predictions(y_test, full_preds)
#         #     self._score_partial_predictions(y_test, full_preds, partial_preds)

#         #     full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 \
#         #         else np.empty((n_samples, full_preds.shape[1]))
#         #     full_preds_n[:] = np.nan
#         #     full_preds_n[test_indices] = full_preds
#         #     self._full_preds = full_preds_n
#         # self.is_fitted = True











# #  self.lfi_abs = lfi_abs
# #         if self.local_scoring_fns and self.mode == "keep_rest":
# #             raise ValueError("Local feature importances have not yet been implemented when mode='keep_rest'.")
# #         if not isinstance(self.local_scoring_fns, bool):
# #             if isinstance(self.scoring_fns, dict):
# #                 if not isinstance(self.local_scoring_fns, dict):
# #                     raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
# #                 for fn_name in self.scoring_fns.keys():
# #                     if fn_name not in self.local_scoring_fns.keys():
# #                         raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
# #             else:
# #                 if not callable(self.local_scoring_fns):
# #                     raise ValueError("local_scoring_fns must be a boolean or a function (given that scoring_fns is not a dictionary).")