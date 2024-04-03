import copy
import numpy as np
import pandas as pd
import shap
import lime
import pprint
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import reduce
from joblib import Parallel, delayed
from imodels.tree.rf_plus.data_transformations.block_transformers import BlockPartitionedData, _blocked_train_test_split
from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices


def _fast_r2_score(y_true, y_pred, multiclass=False):
    """
    Evaluates the r-squared value between the observed and estimated responses.
    Equivalent to sklearn.metrics.r2_score but without the robust error
    checking, thus leading to a much faster implementation (at the cost of
    this error checking). For multi-class responses, returns the mean
    r-squared value across each column in the response matrix.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted responses.
    multiclass: bool
        Whether or not the responses are multi-class.

    Returns
    -------
    Scalar quantity, measuring the r-squared value.
    """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
        sum(axis=0, dtype=np.float64)
    if multiclass:
        return np.mean(1 - numerator / denominator)
    else:
        return 1 - numerator / denominator


def _neg_log_loss(y_true, y_pred):
    """
    Evaluates the negative log-loss between the observed and
    predicted responses.

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Observed responses.
    y_pred: array-like of shape (n_samples, n_targets)
        Predicted probabilies.

    Returns
    -------
    Scalar quantity, measuring the negative log-loss value.
    """
    return -log_loss(y_true, y_pred)

def _get_kernel_shap_rf_plus(model_pred_func,task,X_train,X_test,p,use_summary,k,random_state):
    """
    Get the SHAP values for the RF+ model.

    Parameters
    ----------
    model: RandomForestPlusRegressor or RandomForestPlusClassifier
        The RF+ model.
    X_train: array-like of shape (n_samples, n_features)
        Training data.
    X_test: array-like of shape (n_samples, n_features)
        Testing data.
    p: int
        Number of features to use in the SHAP values.
    use_summary: bool
        Whether or not to use the summary statistics.
    k: int
        Number of cluster to use in the SHAP.kmeans function.

    Returns
    -------
    SHAP values.
    """
    # check that p is a valid proportion
    assert 0 < p <= 1, "p must be in the interval (0, 1]"
        
    # get the subset of the training data to use
    np.random.seed(random_state)

    def add_abs(a, b):
        return abs(a) + abs(b)
        
    # for faster computation, we may want to use shap.kmeans
    if use_summary:
        X_train_summary = shap.kmeans(X_train, k)
    else:
        X_train_subset = shap.utils.sample(X_train,int(p * X_train.shape[0]))
    
    # fit the KernelSHAP model and get the SHAP values
    ex = shap.KernelExplainer(model_pred_func, X_train_summary)
    
    
    shap_values = ex.shap_values(X_test,l1_reg = "num_features(10)")


    if task == "classification":
        shap_values = np.sum(np.abs(shap_values),axis=-1)
    else:
        shap_values = abs(shap_values)
    return shap_values
      

          #shap_values = ex.shap_values(X_test)

def _get_lime_scores_rf_plus(model_pred_func, task, X_train,X_test,random_state) -> np.ndarray:
        """
        Obtain LIME feature importances.

        Inputs:
            X_train (np.ndarray): The training covariate matrix. This is
                                 necessary to fit the LIME model.
            X_test (np.ndarray): The testing covariate matrix. This is the data
                                the resulting LIME values will be based on.
            num_samples (int): The number of samples to use when fitting the
                               LIME model.
        """
        
        # set seed for reproducibility
        np.random.seed(random_state)
        
        # get shape of X_test
        num_samples, num_features = X_test.shape
        
        # create data structure to save scores in
        result = np.zeros((num_samples, num_features))
        
        # initialize the LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train,verbose=False,mode=task)
        for i in range(num_samples):
            exp = explainer.explain_instance(X_test[i,:], model_pred_func,
                                             num_features=num_features)
            original_feature_importance = exp.as_map()[1]
            sorted_feature_importance = sorted(original_feature_importance,
                                               key = lambda x: x[0])
            for j in range(num_features):
                result[i,j] = abs(sorted_feature_importance[j][1])
        # Convert the array to a DataFrame
        lime_values = pd.DataFrame(result, columns=[f'Feature_{i}' for \
                                       i in range(num_features)])

        return lime_values

def _check_Xy(X,y):
    if isinstance(X, pd.DataFrame):
        feature_names_ = list(X.columns)
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        raise ValueError("Input X must be a pandas DataFrame or numpy array.")
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    elif not isinstance(y, np.ndarray):
        raise ValueError("Input y must be a pandas DataFrame or numpy array.")
    return X_array, y

def _check_X(X):
    if isinstance(X, pd.DataFrame):
        feature_names_ = list(X.columns)
        X_array = X.values
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        raise ValueError("Input X must be a pandas DataFrame or numpy array.")
    return X_array


def _tensorize_data(X,transformers):
    """
    Tensorize the data for RF+ model.
    if tokenization_scheme is "feature", then the data is tokenized by feature.
    if tokenization_scheme is "Tree", then the data is tokenized by tree.
    """
    tensorized_data = []
    for transformer in transformers:
        X_transformed = transformer.transform(X) #type: BlockPartitionedData
        max_embedding_size = max(block.shape[1] for block in X_transformed._data_blocks)           
        padded_matrices = [np.pad(matrix, ((0, 0), (0, max_embedding_size - matrix.shape[1])), mode='constant', constant_values=np.nan) for matrix in X_transformed._data_blocks]
        padded_matrices = np.stack(padded_matrices, axis=1)
        tensorized_data.append(padded_matrices) #each tree is one element 
    return tensorized_data


def _tensorize_data_by_tree(X,transformers):
    tensorized_data = []
    max_embedding_size = -1
    for transformer in transformers:
        X_transformed = transformer.transform(X).get_all_data() 
        if X_transformed.shape[1] > max_embedding_size:
            max_embedding_size = X_transformed.shape[1]
        tensorized_data.append(X_transformed) 
    tensorized_data = [np.pad(matrix, ((0, 0), (0, max_embedding_size - matrix.shape[1])), mode='constant', constant_values=np.nan) for matrix in tensorized_data]
    tensorized_data = np.stack(tensorized_data, axis=1)
    return tensorized_data

def _get_sample_split_data(blocked_data, y, random_state):
    in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, in_bag_indices, oob_indices = _blocked_train_test_split(blocked_data,y,random_state)
    return in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, in_bag_indices,oob_indices


    