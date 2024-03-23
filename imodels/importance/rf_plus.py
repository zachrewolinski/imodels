import numpy as np
import pandas as pd
import copy, pprint
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imodels.importance.block_transformers import MDIPlusDefaultTransformer, TreeTransformer, CompositeTransformer, IdentityTransformer
from imodels.importance.ppms import RidgeRegressorPPM, LogisticClassifierPPM, GlmClassifierPPM, GlmRegressorPPM
from imodels.importance.mdi_plus import ForestMDIPlus, _get_sample_split_data
from imodels.importance.rf_plus_utils import _fast_r2_score, _neg_log_loss, _get_kernel_shap_rf_plus, _get_lime_scores_rf_plus, _check_X, _check_Xy
from imodels.importance.ppms_base import *
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.datasets import load_breast_cancer, load_diabetes

class _RandomForestPlus(BaseEstimator):
    """
    The class object for the Random Forest Plus (RF+) estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].

    Parameters
    ----------
    rf_model: scikit-learn random forest object or None
        The RF model to be used to build RF+. If None, then a new
        RandomForestRegressor() or RandomForestClassifier() is instantiated.
    prediction_model: A PartialPredictionModelBase object, scikit-learn type estimator, or "auto"
        The prediction model to be used for making (full and partial) predictions
        using the block transformed data.
        If value is set to "auto", then a default is chosen as follows:
         - For RandomForestPlusRegressor, RidgeRegressorPPM is used.
         - For RandomForestPlusClassifier, LogisticClassifierPPM is used.
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used for fitting RF+. If "oob",
        RF+ is trained on the out-of-bag data. If "inbag", RF+ is trained on the
        in-bag data. Otherwise, RF+ is trained on the full training data.
    include_raw: bool
        Flag for whether to augment the local decision stump features extracted
        from the RF model with the original features.
    drop_features: bool
        Flag for whether to use an intercept model for the partial predictions
        on a given feature if a tree does not have any nodes that split on it,
        or to use a model on the raw feature (if include_raw is True).
    add_transformers: list of BlockTransformerBase objects or None
        Additional block transformers (if any) to include in the RF+ model.
    center: bool
        Flag for whether to center the transformed data in the transformers.
    normalize: bool
        Flag for whether to rescale the transformed data to have unit
        variance in the transformers.
    """

    def __init__(self, rf_model=None, prediction_model=None,
                 include_raw=True, drop_features=True, add_transformers=None, 
                 center=True, normalize=False, cv_ridge = None, n_jobs = -1, 
                fit_on = "auto",choose_reg_param = "loo",multi_task = True):
        
        assert fit_on in ["inbag","oob","all","auto"]
        assert multi_class in ["auto","ovr","multinomial"], "multi_class must be either 'auto', 'ovr', or 'multinomial'"
        assert (choose_reg_param == "loo") or (isinstance(choose_reg_param, int) and choose_reg_param >= 0), "choose_reg_param must be either 'loo' or a non-negative integer"
        # Rest of the function code
        super().__init__()
        
        if isinstance(self, RegressorMixin):
            self._task = "regression"
        elif isinstance(self, ClassifierMixin):
            self._task = "classification"
        else:
            raise ValueError("Unknown task.")
        if rf_model is None:
            if self._task == "regression":
                rf_model = RandomForestRegressor()
            elif self._task == "classification":
                rf_model = RandomForestClassifier()
        if prediction_model is None:
            if self._task == "regression":
                prediction_model = RidgeRegressorPPM(loo=choose_reg_param)
            elif self._task == "classification":
                prediction_model = LogisticClassifierPPM(loo=choose_reg_param)
        
        
        self.rf_model = rf_model
        self.random_state = rf_model.random_state
        self.prediction_model = prediction_model
        self.include_raw = include_raw
        self.drop_features = drop_features
        self.add_transformers = add_transformers
        self.center = center
        self.normalize = normalize
        
       
        if fit_on == "auto":
            self.fit_on = "all"
        else:
            self.fit_on = fit_on
        self.choose_reg_param = choose_reg_param
        self.n_jobs = n_jobs
        self._is_ppm = isinstance(prediction_model, PartialPredictionModelBase)

        if hasattr(self.prediction_model, 'multi_task'):
            multi_task = False
        else:
            multi_task = multi_task
        self.multi_task = multi_task
    
        self.transformers_ = []
        self.estimators_ = []
        self._tree_random_states = []
        self.prediction_score_ = None
        self.mdi_plus_ = None
        self.mdi_plus_scores_ = None
        self.feature_names_ = None

    def fit(self, X, y, center=True, sample_weight=None, **kwargs):
        """
        Fit (or train) Random Forest Plus (RF+) prediction model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.
        sample_weight: array-like of shape (n_samples,) or None
            Sample weights to use in random forest fit.
            If None, samples are equally weighted.
        **kwargs:
            Additional arguments to pass to self.prediction_model.fit()

        """
        
        self._n_samples_train = X.shape[0]

        X_array, y = _check_Xy(X, y)
        
        # center data before fitting random forest
        if center:
            X_array = X_array - X_array.mean(axis=0)

        # fit random forest
        n_samples = X_array.shape[0]
        
        # check if self.rf_model has already been fit
        if not hasattr(self.rf_model, "estimators_"):
            self.rf_model.fit(X_array, y, sample_weight=sample_weight)
        
        # onehot encode multiclass response for GlmClassiferPPM
        # if isinstance(self.prediction_model, GlmClassifierPPM):
        #     self._multi_class = False
        #     if len(np.unique(y)) > 2:
        #         self._multi_class = True
        #         self._y_encoder = OneHotEncoder()
        #         y = self._y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        
        # fit model for each tree
        
        
        all_full_preds = []

        #fit GLM on each tree
        def parallel_fit_helper(tree_model):
            if self.add_transformers is None:
                if self.include_raw:
                    transformer = MDIPlusDefaultTransformer(tree_model, drop_features=self.drop_features)
                else:
                    transformer = TreeTransformer(tree_model)
            else:
                if self.include_raw:
                    base_transformer_list = [TreeTransformer(tree_model), IdentityTransformer()]
                else:
                    base_transformer_list = [TreeTransformer(tree_model)]
                transformer = CompositeTransformer(base_transformer_list + self.add_transformers,
                                                   drop_features=self.drop_features)
            blocked_data = transformer.fit_transform(X_array, center=self.center, normalize=self.normalize)
            
        
            in_bag_blocked_data, oob_blocked_data, y_in_bag, y_oob, inbag_indices, oob_indices = _get_sample_split_data(blocked_data, y, tree_model.random_state)
            pred_func = self._get_pred_func()
            if in_bag_blocked_data.get_all_data().shape[1] != 0:  # if tree has >= 1 split
                if self.fit_on == "inbag":
                    self.prediction_model.fit(in_bag_blocked_data.get_all_data(), y_in_bag, **kwargs)
                    full_preds = pred_func(oob_blocked_data.get_all_data())
                    full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 else np.empty((n_samples, full_preds.shape[1]))
                    full_preds_n[:] = np.nan
                    full_preds_n[oob_indices] = full_preds
                elif self.fit_on == "oob":
                    self.prediction_model.fit(oob_blocked_data.get_all_data(), y_oob, **kwargs)
                    full_preds = pred_func(oob_blocked_data.get_all_data())
                    full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 else np.empty((n_samples, full_preds.shape[1]))
                    full_preds_n[:] = np.nan
                    full_preds_n[oob_indices] = full_preds
                else:
                    self.prediction_model.fit(blocked_data.get_all_data(), y, **kwargs)
                    full_preds = pred_func(blocked_data.get_all_data())
                    full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 else np.empty((n_samples, full_preds.shape[1]))
                    full_preds_n[:] = np.nan
                    full_preds_n[:] = full_preds
                return copy.deepcopy(self.prediction_model), copy.deepcopy(transformer), tree_model.random_state,full_preds_n
            else:
                pprint.pprint("Tree has no splits!")
                return None,None,None,None 


        #get prediction error 
        results = Parallel(n_jobs=self.n_jobs)(delayed(parallel_fit_helper)(tree_model) for tree_model in self.rf_model.estimators_)
        for result in results:
            
            self.estimators_.append(result[0])
            self.transformers_.append(result[1])
            self._tree_random_states.append(result[2])
            all_full_preds.append(result[3])
      
        # compute prediction accuracy on internal sample split
        full_preds = np.nanmean(all_full_preds, axis=0)
        if self._task == "regression":
            pred_score = r2_score(y, full_preds)
            pred_score_name = "r2"
        elif self._task == "classification":
            if full_preds.shape[1] == 2:
                pred_score = roc_auc_score(y, full_preds[:, 1], multi_class="ovr")
            else:
                pred_score = roc_auc_score(y, full_preds, multi_class="ovr")
            pred_score_name = "auroc"
        self.prediction_score_ = pd.DataFrame({pred_score_name: [pred_score]})
        self._full_preds = full_preds
        
    

    def predict(self, X):
        """
        Make predictions on new data using the fitted model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples,) or (n_samples, n_targets)
            The predicted values

        """
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        X_array = _check_X(X)

        def parallel_predict_helper(estimator, transformer, data):
            blocked_data = transformer.transform(data, center=self.center, normalize=self.normalize)
            predictions = estimator.predict(blocked_data.get_all_data())
            return predictions

        if self._task == "regression":
            predictions = Parallel(n_jobs=self.n_jobs)(delayed(parallel_predict_helper)(self.estimators_[i], self.transformers_[i], X_array) 
                                                       for i in range(len(self.estimators_)))
            predictions = np.mean(predictions, axis=0)
        elif self._task == "classification":
            prob_predictions = self.predict_proba(X_array)
            if prob_predictions.ndim == 1:
                prob_predictions = np.stack([1-prob_predictions, prob_predictions], axis=1)
            predictions = self.rf_model.classes_[np.argmax(prob_predictions, axis=1)]
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities on new data using the fitted
        (classification) model.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix, for which to make predictions.

        Returns
        -------
        y: ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.

        """
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                self.estimators_[0].__class__.__name__)
            )
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X_array = X.loc[:, self.feature_names_].values
            else:
                X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise ValueError("Input X must be a pandas DataFrame or numpy array.")
        
        def parallel_predict_proba_helper(estimator, transformer, data):
            blocked_data = transformer.transform(data, center=self.center, normalize=self.normalize)
            predictions = estimator.predict_proba(blocked_data.get_all_data())
            return predictions

        predictions = Parallel(n_jobs=self.n_jobs)(delayed(parallel_predict_proba_helper)(self.estimators_[i], self.transformers_[i], X_array)
                                                   for i in range(len(self.estimators_)))
        predictions = np.mean(np.stack(predictions), axis=0)
        return predictions
        
        
        # for estimator, transformer in zip(self.estimators_, self.transformers_):
        #     blocked_data = transformer.transform(X_array, center=self.center, normalize=self.normalize)
        #     predictions += estimator.predict_proba(blocked_data.get_all_data())
        # predictions = predictions / len(self.estimators_)
        # return predictions
    
    def get_kernel_shap_scores(self, X_train: np.ndarray, X_test: np.ndarray,
                               p: float = 1, use_summary: bool = True,
                               k : int = 1) -> np.ndarray:
        """
        Obtain KernelSHAP feature importances.

        Inputs:
            X_train (np.ndarray): The training covariate matrix. This is
                                 necessary to fit the KernelSHAP model.
            X_test (np.ndarray): The testing covariate matrix. This is the data
                                the resulting SHAP values will be based on.
            p (float): The proportion of the training data which will be used to
                       fit the KernelSHAP model. Due to the expensive
                       computation of KernelSHAP, for large datasets it may be
                       helpful to have p < 1.
            use_summary (bool): Whether to use the summary of the SHAP values
                                via shap.kmeans
            k (int): The number of clusters to use for the shap.kmeans algorithm
        """
        if self._task == "regression": 
            model_pred_func = self.predict 
        else: 
            model_pred_func = self.predict_proba
        return _get_kernel_shap_rf_plus(model_pred_func,self._task,X_train,X_test,p,use_summary,k, self.random_state)

    def get_lime_scores(self, X_train: np.ndarray, X_test: np.ndarray):

        if self._task == "regression": 
            model_pred_func = self.predict
        else:
            model_pred_func = self.predict_proba
        return _get_lime_scores_rf_plus(model_pred_func,self._task,X_train,X_test,self.random_state)
    
    
    
    
    def get_local_mdi_plus(self, X=None, y=None, lfi_abs="none",
                           train_or_test="train"):
        X_array, y = self._check_data(X, y)
        # compute MDI+ for forest
        mdi_plus_obj = ForestMDIPlus(estimators=self.estimators_,
                                        transformers=self.transformers_,
                                        scoring_fns=None,
                                        tree_random_states=self._tree_random_states,
                                        task=self._task,
                                        center=self.center,
                                        normalize=self.normalize)
        self.mdi_plus_ = mdi_plus_obj
        mdi_plus_scores = mdi_plus_obj.get_scores(X_array, y, lfi=True,
                                                  lfi_abs=lfi_abs,
                                                  train_or_test=train_or_test)

    def get_mdi_plus_scores(self, X=None, y=None,
                            scoring_fns="auto", local_scoring_fns=False,
                            sample_split="inherit", mode="keep_k",
                            version="all", lfi=False, lfi_abs="inside",
                            train_or_test="train"):
        """
        Obtain MDI+ feature importances. Generalized mean decrease in impurity (MDI+)
        is a flexible framework for computing RF feature importances. For more
        details, refer to [paper].

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. Generally should be the same X as that used for
            fitting the RF+ prediction model. If a pd.DataFrame object is supplied, then
            the column names are used in the output.
        y: ndarray of shape (n_samples, n_targets)
            The observed responses. Generally should be the same y as that used for
            fitting the RF+ prediction model.
        scoring_fns: a function or dict with functions as value and function name (str) as key or "auto"
            The scoring functions used for evaluating the partial predictions (globally).
            If "auto", then a default is chosen as follows:
             - For RandomForestPlusRegressor, then r-squared (_fast_r2_score) is used.
             - For RandomForestPlusClassifier, then the negative log-loss (_neg_log_loss) is used.
        local_scoring_fns: one of True, False, "auto", function or dict with functions as value and function name (str)
            as key. The local scoring functions used for evaluating the partial predictions per sample.
            If False, then local feature importances are not evaluated.
            If True or "auto", then the (global) scoring functions are used as the local scoring functions.
            If a function is provided, this function is used as the local scoring function and applied per sample.
            Otherwise, a dictionary of local scoring functions can be supplied, with one local scoring function per
            global scoring function (using the same key).
        sample_split: string in {"loo", "oob", "inbag", "inherit"} or None
            The sample splitting strategy to be used when evaluating the partial
            model predictions in MDI+. If "inherit" (default), uses the same sample splitting
            strategy as used when fitting the RF+ prediction model. Assuming None or "loo" were used
            as the sample splitting scheme for fitting the RF+ prediction model,
            "loo" (leave-one-out) is strongly recommended here in MDI+ as it overcomes
            the known correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
            also be used to overcome these biases. "inbag" is the sample splitting
            strategy used by MDI. If None, no sample splitting is performed and the
            full data set is used to evaluate the partial model predictions.
        mode: string in {"keep_k", "keep_rest"}
            Mode for the method. "keep_k" imputes the mean of each feature not
            in block k when making a partial model prediction, while "keep_rest"
            imputes the mean of each feature in block k. "keep_k" is strongly
            recommended for computational considerations.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The MDI+ feature importances.
        """
        assert version == "all" or version == "sub"
        if X is None or y is None:
            if self.mdi_plus_scores_ is None:
                raise ValueError("Need X and y as inputs.")
            if local_scoring_fns:
                if self.mdi_plus_scores_local_ is None:
                    raise ValueError("Need X and y as inputs.")
        else:
            # convert data frame to array
            if isinstance(X, pd.DataFrame):
                if self.feature_names_ is not None:
                    X_array = X.loc[:, self.feature_names_].values
                else:
                    X_array = X.values
            elif isinstance(X, np.ndarray):
                X_array = X
            else:
                raise ValueError("Input X must be a pandas DataFrame or numpy array.")
            if isinstance(y, pd.DataFrame):
                y = y.values.ravel()
            elif not isinstance(y, np.ndarray):
                raise ValueError("Input y must be a pandas DataFrame or numpy array.")
            # get defaults
            if sample_split == "inherit":
                sample_split = self.sample_split
            if X.shape[0] != self._n_samples_train and sample_split is not None:
                raise ValueError("Set sample_split=None to fit MDI+ on non-training X and y. "
                                 "To use other sample_split schemes, input the training X and y data.")
            if scoring_fns == "auto":
                scoring_fns = _fast_r2_score if self._task == "regression" \
                    else _neg_log_loss
            if local_scoring_fns:
                if local_scoring_fns == "auto" or local_scoring_fns is True:
                    local_scoring_fns = scoring_fns
                else:
                    if isinstance(local_scoring_fns, dict):
                        for fn_name in scoring_fns.keys():
                            if fn_name not in local_scoring_fns.keys():
                                raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
                    elif not callable(local_scoring_fns):
                        raise ValueError("local_scoring_fns must be a boolean, 'auto', a function, or dictionary of functions.")
            # onehot encode if multi-class for GlmClassiferPPM
            if isinstance(self.prediction_model, GlmClassifierPPM):
                if self._multi_class:
                    y = self._y_encoder.transform(y.reshape(-1, 1)).toarray()
            # compute MDI+ for forest
            mdi_plus_obj = ForestMDIPlus(estimators=self.estimators_,
                                         transformers=self.transformers_,
                                         scoring_fns=scoring_fns,
                                         local_scoring_fns=local_scoring_fns,
                                         sample_split=sample_split,
                                         tree_random_states=self._tree_random_states,
                                         mode=mode,
                                         task=self._task,
                                         center=self.center,
                                         normalize=self.normalize,
                                         version=version)
            self.mdi_plus_ = mdi_plus_obj
            mdi_plus_scores = mdi_plus_obj.get_scores(X_array, y, lfi=lfi,
                                                      lfi_abs=lfi_abs,
                                                      train_or_test=train_or_test)
            if lfi and local_scoring_fns:
                mdi_plus_lfi = mdi_plus_scores["lfi"]
                mdi_plus_scores_local = mdi_plus_scores["local"]
                mdi_plus_scores = mdi_plus_scores["global"]
                self.mdi_plus_lfi = mdi_plus_lfi
            if lfi and (not local_scoring_fns):
                mdi_plus_lfi = mdi_plus_scores["lfi"]
                mdi_plus_scores = mdi_plus_scores["global"]
                self.mdi_plus_lfi = mdi_plus_lfi
            if (not lfi) and local_scoring_fns:
                mdi_plus_scores_local = mdi_plus_scores["local"]
                mdi_plus_scores = mdi_plus_scores["global"]
            if self.feature_names_ is not None:
                mdi_plus_scores["var"] = self.feature_names_
                self.mdi_plus_.feature_importances_["var"] = self.feature_names_
                if local_scoring_fns:
                    mdi_plus_scores_local["var"] = self.feature_names_
                    self.mdi_plus_.feature_importances_local_["var"] = self.feature_names_
            self.mdi_plus_scores_ = mdi_plus_scores
            if local_scoring_fns:
                self.mdi_plus_scores_local_ = mdi_plus_scores_local
        if lfi:
            if local_scoring_fns:
                return {"global": self.mdi_plus_scores_,
                        "local": self.mdi_plus_scores_local_,
                        "lfi": self.mdi_plus_lfi}
            else:
                return {"global": self.mdi_plus_scores_,
                        "lfi": self.mdi_plus_lfi}
        if local_scoring_fns:
            return {"global": self.mdi_plus_scores_,
                    "local": self.mdi_plus_scores_local_}
        else:
            return self.mdi_plus_scores_

    def get_mdi_plus_stability_scores(self, B=10, metrics="auto"):
        """
        Evaluate the stability of the MDI+ feature importance rankings
        across bootstrapped samples of trees. Can be used to select the GLM
        and scoring metric in a data-driven manner, where the GLM and metric, which
        yields the most stable feature rankings across bootstrapped samples is selected.

        Parameters
        ----------
        B: int
            Number of bootstrap samples.
        metrics: "auto" or a dict with functions as value and function name (str) as key
            Metric(s) used to evaluate the stability between two sets of feature importances.
            If "auto", then the feature importance stability metrics are:
                (1) Rank-based overlap (RBO) with p=0.9 (from "A Similarity Measure for
                Indefinite Rankings" by Webber et al. (2010)). Intuitively, this metric gives
                more weight to features with the largest importances, with most of the weight
                going to the ~1/(1-p) features with the largest importances.
                (2) A weighted kendall tau metric (tauAP_b from "The Treatment of Ties in
                AP Correlation" by Urbano and Marrero (2017)), which also gives more weight
                to the features with the largest importances, but uses a different weighting
                scheme from RBO.
            Note that these default metrics assume that a higher MDI+ score indicates
            greater importance and thus give more weight to these features with high
            importance/ranks. If a lower MDI+ score indicates higher importance, then invert
            either these stability metrics or the MDI+ scores before evaluating the stability.

        Returns
        -------
        stability_results: pd.DataFrame of shape (n_features, n_metrics)
            The stability scores of the MDI+ feature rankings across bootstrapped samples.

        """
        if self.mdi_plus_ is None:
            raise ValueError("Need to compute MDI+ scores first using self.get_mdi_plus_scores(X, y)")
        return self.mdi_plus_.get_stability_scores(B=B, metrics=metrics)

    def _get_pred_func(self):
        if hasattr(self.prediction_model, "predict_proba_loo"):
            pred_func = self.prediction_model.predict_proba_loo
        elif hasattr(self.prediction_model, "predict_loo"):
            pred_func = self.prediction_model.predict_loo
        elif hasattr(self.prediction_model, "predict_proba"):
            pred_func = self.prediction_model.predict_proba
        else:
            pred_func = self.prediction_model.predict
        return pred_func
    
    def _check_data(self, X=None, y=None):
        if X is None or y is None:
            if self.mdi_plus_scores_ is None:
                raise ValueError("Need X and y as inputs.")
        else:
            # convert data frame to array
            if isinstance(X, pd.DataFrame):
                if self.feature_names_ is not None:
                    X_array = X.loc[:, self.feature_names_].values
                else:
                    X_array = X.values
            elif isinstance(X, np.ndarray):
                X_array = X
            else:
                raise ValueError("Input X must be a pandas DataFrame or numpy array.")
            if isinstance(y, pd.DataFrame):
                y = y.values.ravel()
            elif not isinstance(y, np.ndarray):
                raise ValueError("Input y must be a pandas DataFrame or numpy array.")            
            # onehot encode if multi-class for GlmClassiferPPM
            if isinstance(self.prediction_model, GlmClassifierPPM):
                if self._multi_class:
                    y = self._y_encoder.transform(y.reshape(-1, 1)).toarray()
        return X_array, y        


class RandomForestPlusRegressor(_RandomForestPlus, RegressorMixin):
    """
    The class object for the Random Forest Plus (RF+) regression estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    def __init__(self, rf_model=None, prediction_model=None, include_raw=True, drop_features=True, 
                 add_transformers=None, center=True, normalize=False, cv_ridge=None, n_jobs=-1,
                 fit_on="auto", choose_reg_param="loo",multi_task = "auto"):
        super().__init__(rf_model, prediction_model, include_raw, drop_features, add_transformers, center, normalize, cv_ridge, n_jobs, fit_on, choose_reg_param, )


class RandomForestPlusClassifier(_RandomForestPlus, ClassifierMixin):
    """
    The class object for the Random Forest Plus (RF+) classification estimator, which can
    be used as a prediction model or interpreted via generalized
    mean decrease in impurity (MDI+). For more details, refer to [paper].
    """
    ...




if __name__ == "__main__":


    test_size = 0.2
    n_jobs = 1
    
    
    #test regression
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1,)

    pprint.pprint(f"Shape: {X_train.shape}")

    rf = RandomForestRegressor(n_estimators=3,min_samples_leaf=5,max_features=0.33,random_state=1)
    rf.fit(X_train, y_train)
    pprint.pprint(f"RF r2_score: {r2_score(y_test,rf.predict(X_test))}")

    rf_plus = RandomForestPlusRegressor(rf_model=copy.deepcopy(rf),n_jobs = n_jobs)
    rf_plus.fit(X_train, y_train)
    pprint.pprint(f"RF+ r2_score: {r2_score(y_test,rf_plus.predict(X_test))}")



    #test get regression shap scores
    #shap_values = rf_plus.get_kernel_shap_scores(X_train, X_test) 
    #test get regression LIME scores
    #lime_values = rf_plus.get_lime_scores(X_train, X_test)

    # #test classification
    # X, y = load_breast_cancer(return_X_y=True)
    # print(X_test.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    # rf_clf = RandomForestClassifier(n_estimators=3,min_samples_leaf=1,max_features="sqrt",random_state=1)
    # rf_clf.fit(X_train, y_train)
    # pprint.pprint(f"RF roc_auc_score: {roc_auc_score(y_test,rf_clf.predict_proba(X_test)[:,1])}")
    # rf_plus_clf = RandomForestPlusClassifier(rf_model=copy.deepcopy(rf_clf),n_jobs = n_jobs)
    # rf_plus_clf.fit(X_train, y_train)
    # pprint.pprint(f"RF+ roc_auc_score: {roc_auc_score(y_test,rf_plus_clf.predict_proba(X_test)[:,1])}")

    # #test get classification shap scores
    # shap_values = rf_plus_clf.get_kernel_shap_scores(X_train, X_test)
    # pprint.pprint(shap_values.shape)
    # #test get classification LIME scores
    # lime_values = rf_plus_clf.get_lime_scores(X_train, X_test)
    # pprint.pprint(lime_values.shape)

    


    #pprint.pprint(shap_values.shape)
   

    #test get local scores
    











