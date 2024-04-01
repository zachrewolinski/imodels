import numpy as np
import pandas as pd
import  pprint, copy, imodels
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imodels.tree.rf_plus.data_transformations.block_transformers import MDIPlusDefaultTransformer, TreeTransformer, CompositeTransformer, IdentityTransformer
from imodels.tree.rf_plus.ppms.ppms import PartialPredictionModelBase, GlmClassifierPPM, RidgeRegressorPPM, LogisticClassifierPPM
from imodels.tree.rf_plus.ppms.ppms_regression import GlmNetElasticNetRegressorPPM, GlmNetRidgeRegressorPPM, GlmNetLassoRegressorPPM
from imodels.tree.rf_plus.ppms.ppms_classification import GlmClassifierPPM, GLMLogisticElasticNetPPM
from imodels.tree.rf_plus.rf_plus_utils import _fast_r2_score, _neg_log_loss, _get_kernel_shap_rf_plus, _get_lime_scores_rf_plus, _check_X, _check_Xy, _tensorize_data, _tensorize_data_by_tree, _get_sample_split_data
from glmnet import ElasticNet, LogitNet
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices



     
