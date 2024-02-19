import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from functools import partial

from .ppms import PartialPredictionModelBase, GenericRegressorPPM, GenericClassifierPPM, _extract_coef_and_intercept
from .block_transformers import _blocked_train_test_split
from .ranking_stability import tauAP_b, rbo


class ForestMDIPlus:
    """
    The class object for computing MDI+ feature importances for a forest or collection of trees.
    Generalized mean decrease in impurity (MDI+) is a flexible framework for computing RF
    feature importances. For more details, refer to [paper].

    Parameters
    ----------
    estimators: list of fitted PartialPredictionModelBase objects or scikit-learn type estimators
        The fitted partial prediction models (one per tree) to use for evaluating
        feature importance via MDI+. If not a PartialPredictionModelBase, then
        the estimator is coerced into a PartialPredictionModelBase object via
        GenericRegressorPPM or GenericClassifierPPM depending on the specified
        task. Note that these generic PPMs may be computationally expensive.
    transformers: list of BlockTransformerBase objects
        The block feature transformers used to generate blocks of engineered
        features for each original feature. The transformed data is then used
        as input into the partial prediction models. Should be the same length
        as estimators.
    scoring_fns: a function or dict with functions as value and function name (str) as key
        The scoring functions used for evaluating the partial predictions.
    local_scoring_fns: one of True, False, function or dict with functions as value and function name (str) as key.
        The local scoring functions used for evaluating the partial predictions per sample.
        If False, then local feature importances are not evaluated.
        If True, then the (global) scoring functions are used as the local scoring functions.
        If a function is provided, this function is used as the local scoring function and applied per sample.
        Otherwise, a dictionary of local scoring functions can be supplied, with one local scoring function per
        global scoring function (using the same key).
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used when evaluating the partial
        model predictions. The default "loo" (leave-one-out) is strongly
        recommended for performance and in particular, for overcoming the known
        correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
        also be used to overcome these biases. "inbag" is the sample splitting
        strategy used by MDI. If None, no sample splitting is performed and the
        full data set is used to evaluate the partial model predictions.
    tree_random_states: list of int or None
        Random states from each tree in the fitted random forest; used in
        sample splitting and only required if sample_split = "oob" or "inbag".
        Should be the same length as estimators.
    mode: string in {"keep_k", "keep_rest"}
        Mode for the method. "keep_k" imputes the mean of each feature not
        in block k when making a partial model prediction, while "keep_rest"
        imputes the mean of each feature in block k. "keep_k" is strongly
        recommended for computational considerations.
    task: string in {"regression", "classification"}
        The supervised learning task for the RF model. Used for choosing
        defaults for the scoring_fns. Currently only regression and
        classification are supported.
    center: bool
        Flag for whether to center the transformed data in the transformers.
    normalize: bool
        Flag for whether to rescale the transformed data to have unit
        variance in the transformers.
    """

    def __init__(self, estimators, transformers, scoring_fns, local_scoring_fns=False,
                 sample_split="loo", tree_random_states=None, mode="keep_k",
                 task="regression", center=True, normalize=False,version="all"):
        assert version == "all" or version == "sub"
        assert sample_split in ["loo", "oob", "inbag", None]
        assert mode in ["keep_k", "keep_rest"]
        assert task in ["regression", "classification"]
        # print("CREATING FOREST MDI PLUS OBJECT")
        self.estimators = estimators
        self.transformers = transformers
        self.scoring_fns = scoring_fns
        self.local_scoring_fns = local_scoring_fns
        self.sample_split = sample_split
        self.tree_random_states = tree_random_states
        if self.sample_split in ["oob", "inbag"] and not self.tree_random_states:
            raise ValueError("Must specify tree_random_states to use 'oob' or 'inbag' sample_split.")
        self.mode = mode
        self.task = task
        self.center = center
        self.normalize = normalize
        self.version = version
        if self.local_scoring_fns and self.mode == "keep_rest":
            raise ValueError("Local feature importances have not yet been implemented when mode='keep_rest'.")
        if not isinstance(self.local_scoring_fns, bool):
            if isinstance(self.scoring_fns, dict):
                if not isinstance(self.local_scoring_fns, dict):
                    raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
                for fn_name in self.scoring_fns.keys():
                    if fn_name not in self.local_scoring_fns.keys():
                        raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
            else:
                if not callable(self.local_scoring_fns):
                    raise ValueError("local_scoring_fns must be a boolean or a function (given that scoring_fns is not a dictionary).")
        self.is_fitted = False
        self.prediction_score_ = pd.DataFrame({})
        self.feature_importances_ = pd.DataFrame({})
        self.feature_importances_by_tree_ = {}
        self.feature_importances_local_ = {}
        self.feature_importances_local_by_tree_ = {}

    def get_scores(self, X, y, lfi=False, lfi_abs="inside"):
        """
        Obtain the MDI+ feature importances for a forest.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. If a pd.DataFrame object is supplied, then
            the column names are used in the output
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The MDI+ feature importances.
        """
        # print("IN 'get_scores' METHOD WITHIN THE FOREST MDI PLUS OBJECT")
        self.lfi_abs = lfi_abs
        self._fit_importance_scores(X, y)
        if lfi:
            if self.local_scoring_fns:
                return {"global": self.feature_importances_,
                        "local": self.feature_importances_local_,
                        "lfi": self.final_lfi_matrix}
            else:
                return {"global": self.feature_importances_,
                        "lfi": self.final_lfi_matrix}
        if self.local_scoring_fns:
            return {"global": self.feature_importances_,
                    "local": self.feature_importances_local_}
        else:
            return self.feature_importances_

    def get_stability_scores(self, B=10, metrics="auto"):
        """
        Evaluate the stability of the MDI+ feature importance rankings
        across bootstrapped samples of trees. Can be used to select the GLM
        and scoring metric in a data-driven manner, where the GLM and metric that
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
        # print("IN 'get_stability_scores' METHOD WITHIN THE FOREST MDI PLUS OBJECT")
        if metrics == "auto":
            metrics = {"RBO": partial(rbo, p=0.9), "tauAP": tauAP_b}
        elif not isinstance(metrics, dict):
            raise ValueError("`metrics` must be 'auto' or a dictionary "
                             "where the key is the metric name and the value is the evaluation function")
        single_scoring_fn = not isinstance(self.feature_importances_by_tree_, dict)
        if single_scoring_fn:
            feature_importances_dict = {"mdi_plus_score": self.feature_importances_by_tree_}
        else:
            feature_importances_dict = self.feature_importances_by_tree_
        stability_dict = {}
        for scoring_fn_name, feature_importances_by_tree in feature_importances_dict.items():
            n_trees = feature_importances_by_tree.shape[1]
            fi_scores_boot_ls = []
            for b in range(B):
                bootstrap_sample = np.random.choice(n_trees, n_trees, replace=True)
                fi_scores_boot_ls.append(feature_importances_by_tree[bootstrap_sample].mean(axis=1))
            fi_scores_boot = pd.concat(fi_scores_boot_ls, axis=1)
            stability_results = {"scorer": [scoring_fn_name]}
            for metric_name, metric_fun in metrics.items():
                stability_results[metric_name] = [np.mean(pdist(fi_scores_boot.T, metric=metric_fun))]
            stability_dict[scoring_fn_name] = pd.DataFrame(stability_results)
        stability_df = pd.concat(stability_dict, axis=0).reset_index(drop=True)
        if single_scoring_fn:
            stability_df = stability_df.drop(columns=["scorer"])
        return stability_df

    def _fit_importance_scores(self, X, y):
        all_scores = []
        all_full_preds = []
        all_local_scores = []
        num_iters = len(list(zip(self.estimators, self.transformers, self.tree_random_states)))
        lfi_matrix_lst = list()
        for estimator, transformer, tree_random_state in \
                zip(self.estimators, self.transformers, self.tree_random_states):
            tree_mdi_plus = TreeMDIPlus(estimator=estimator,
                                        transformer=transformer,
                                        scoring_fns=self.scoring_fns,
                                        local_scoring_fns=self.local_scoring_fns,
                                        sample_split=self.sample_split,
                                        tree_random_state=tree_random_state,
                                        mode=self.mode,
                                        task=self.task,
                                        center=self.center,
                                        normalize=self.normalize,
                                        version=self.version,
                                        num_iters=num_iters,
                                        lfi_abs=self.lfi_abs)
            scores = tree_mdi_plus.get_scores(X, y)
            lfi_matrix_lst.append(tree_mdi_plus.lfi_matrix)
            if scores is not None:
                if self.local_scoring_fns:
                    local_scores = scores["local"]
                    scores = scores["global"]
                    all_local_scores.append(local_scores)
                all_scores.append(scores)
                all_full_preds.append(tree_mdi_plus._full_preds)
        stacked_lfi_matrices = np.stack(lfi_matrix_lst, axis=0)
        average_lfi_matrix = np.mean(stacked_lfi_matrices, axis=0)
        self.final_lfi_matrix = pd.DataFrame(average_lfi_matrix)
        # print("LFI MATRIX")
        # print(pd.DataFrame(self.lfi_matrix))
        if len(all_scores) == 0:
            raise ValueError("Transformer representation was empty for all trees.")
        full_preds = np.nanmean(all_full_preds, axis=0)
        self._full_preds = full_preds
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"importance": self.scoring_fns}
        for fn_name, scoring_fn in scoring_fns.items():
            self.feature_importances_by_tree_[fn_name] = pd.concat([scores[fn_name] for scores in all_scores], axis=1)
            self.feature_importances_by_tree_[fn_name].columns = np.arange(len(all_scores))
            self.feature_importances_[fn_name] = np.mean(self.feature_importances_by_tree_[fn_name], axis=1)
            self.prediction_score_[fn_name] = [scoring_fn(y[~np.isnan(full_preds)], full_preds[~np.isnan(full_preds)])]
            if self.local_scoring_fns:
                self.feature_importances_local_by_tree_[fn_name] = [local_scores[fn_name] for local_scores in all_local_scores]
                self.feature_importances_local_[fn_name] = pd.DataFrame(
                    np.mean(self.feature_importances_local_by_tree_[fn_name], axis=0)
                )
                if isinstance(X, pd.DataFrame):
                    self.feature_importances_local_[fn_name].columns = X.columns
        if list(scoring_fns.keys()) == ["importance"]:
            self.prediction_score_ = self.prediction_score_["importance"]
            self.feature_importances_by_tree_ = self.feature_importances_by_tree_["importance"]
            if self.local_scoring_fns:
                self.feature_importances_local_by_tree_ = self.feature_importances_local_by_tree_["importance"]
                self.feature_importances_local_ = self.feature_importances_local_["importance"]
        if isinstance(X, pd.DataFrame):
            self.feature_importances_.index = X.columns
        self.feature_importances_.index.name = 'var'
        self.feature_importances_.reset_index(inplace=True)
        self.is_fitted = True


class TreeMDIPlus:
    """
    The class object for computing MDI+ feature importances for a single tree.
    Generalized mean decrease in impurity (MDI+) is a flexible framework for computing RF
    feature importances. For more details, refer to [paper].

    Parameters
    ----------
    estimator: a fitted PartialPredictionModelBase object or scikit-learn type estimator
        The fitted partial prediction model to use for evaluating
        feature importance via MDI+. If not a PartialPredictionModelBase, then
        the estimator is coerced into a PartialPredictionModelBase object via
        GenericRegressorPPM or GenericClassifierPPM depending on the specified
        task. Note that these generic PPMs may be computationally expensive.
    transformer: a BlockTransformerBase object
        A block feature transformer used to generate blocks of engineered
        features for each original feature. The transformed data is then used
        as input into the partial prediction models.
    scoring_fns: a function or dict with functions as value and function name (str) as key
        The scoring functions used for evaluating the partial predictions.
    local_scoring_fns: one of True, False, function or dict with functions as value and function name (str) as key.
        The local scoring functions used for evaluating the partial predictions per sample.
        If False, then local feature importances are not evaluated.
        If True, then the (global) scoring functions are used as the local scoring functions.
        If a function is provided, this function is used as the local scoring function and applied per sample.
        Otherwise, a dictionary of local scoring functions can be supplied, with one local scoring function per
        global scoring function (using the same key).
    sample_split: string in {"loo", "oob", "inbag"} or None
        The sample splitting strategy to be used when evaluating the partial
        model predictions. The default "loo" (leave-one-out) is strongly
        recommended for performance and in particular, for overcoming the known
        correlation and entropy biases suffered by MDI. "oob" (out-of-bag) can
        also be used to overcome these biases. "inbag" is the sample splitting
        strategy used by MDI. If None, no sample splitting is performed and the
        full data set is used to evaluate the partial model predictions.
    tree_random_state: int or None
        Random state of the fitted tree; used in sample splitting and
        only required if sample_split = "oob" or "inbag".
    mode: string in {"keep_k", "keep_rest"}
        Mode for the method. "keep_k" imputes the mean of each feature not
        in block k when making a partial model prediction, while "keep_rest"
        imputes the mean of each feature in block k. "keep_k" is strongly
        recommended for computational considerations.
    task: string in {"regression", "classification"}
        The supervised learning task for the RF model. Used for choosing
        defaults for the scoring_fns. Currently only regression and
        classification are supported.
    center: bool
        Flag for whether to center the transformed data in the transformers.
    normalize: bool
        Flag for whether to rescale the transformed data to have unit
        variance in the transformers.
    """

    def __init__(self, estimator, transformer, scoring_fns, local_scoring_fns=False,
                 sample_split="loo", tree_random_state=None, mode="keep_k",
                 task="regression", center=True, normalize=False,version="all",
                 num_iters=-1, lfi_abs="inside"):
        assert version == "all" or version == "sub"
        assert sample_split in ["loo", "oob", "inbag", "auto", None]
        assert mode in ["keep_k", "keep_rest"]
        assert task in ["regression", "classification"]
        # print("CREATING TREE MDI PLUS OBJECT")
        self.estimator = estimator
        self.transformer = transformer
        self.version = version
        self.num_iters = num_iters
        self.scoring_fns = scoring_fns
        self.local_scoring_fns = local_scoring_fns
        self.sample_split = sample_split
        self.tree_random_state = tree_random_state
        _validate_sample_split(self.sample_split, self.estimator, isinstance(self.estimator, PartialPredictionModelBase))
        if self.sample_split in ["oob", "inbag"] and not self.tree_random_state:
            raise ValueError("Must specify tree_random_state to use 'oob' or 'inbag' sample_split.")
        self.mode = mode
        self.task = task
        self.center = center
        self.normalize = normalize
        self.lfi_abs = lfi_abs
        if self.local_scoring_fns and self.mode == "keep_rest":
            raise ValueError("Local feature importances have not yet been implemented when mode='keep_rest'.")
        if not isinstance(self.local_scoring_fns, bool):
            if isinstance(self.scoring_fns, dict):
                if not isinstance(self.local_scoring_fns, dict):
                    raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
                for fn_name in self.scoring_fns.keys():
                    if fn_name not in self.local_scoring_fns.keys():
                        raise ValueError("Since scoring_fns is a dictionary, local_scoring_fns must also be a dictionary with one local scoring function for each scoring function using the same key.")
            else:
                if not callable(self.local_scoring_fns):
                    raise ValueError("local_scoring_fns must be a boolean or a function (given that scoring_fns is not a dictionary).")
        self.is_fitted = False
        self._full_preds = None
        self.prediction_score_ = None
        self.feature_importances_ = None
        self.feature_importances_local_ = None

    def get_scores(self, X, y):
        """
        Obtain the MDI+ feature importances for a single tree.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            The covariate matrix. If a pd.DataFrame object is supplied, then
            the column names are used in the output
        y: ndarray of shape (n_samples, n_targets)
            The observed responses.

        Returns
        -------
        scores: pd.DataFrame of shape (n_features, n_scoring_fns)
            The MDI+ feature importances.
        """
        # print("IN 'get_scores' METHOD WITHIN THE TREE MDI PLUS OBJECT")
        self._fit_importance_scores(X, y)
        if self.local_scoring_fns:
            return {"global": self.feature_importances_,
                    "local": self.feature_importances_local_}
        else:
            return self.feature_importances_

    def _fit_importance_scores(self, X, y):
        # print("IN '_fit_importance_scores' METHOD WITHIN THE TREE MDI PLUS OBJECT")
        n_samples = y.shape[0]
        zero_values = None
        if self.version == "sub":
            blocked_data, zero_values = self.transformer.transform(X,
                                                    center=self.center,
                                                    normalize=self.normalize,
                                                    zeros=True)
        else:
            blocked_data = self.transformer.transform(X,
                                                    center=self.center,
                                                    normalize=self.normalize,
                                                    zeros=False)
        
        coefs = self.estimator.coefficients_
        loo_coefs = self.estimator.loo_coefficients_
        lfi_matrix = np.zeros((blocked_data.get_all_data().shape[0], X.shape[1]))
        for j in range(self.estimator._n_outputs):
            # actually not sure what to do in the case with multiple outputs
            if loo_coefs[j].shape[1] == (blocked_data.get_all_data().shape[1] + 1):
                intercept = loo_coefs[j][:,-1]
                loo_coefs_j = loo_coefs[j][:,:-1]
            else:
                loo_coefs_j = loo_coefs[j]
            coef_idx = 0
            for k in range(blocked_data.n_blocks):
                block_k = blocked_data.get_block(k)
                if self.lfi_abs == "inside":
                    lfi_matrix[:, k] = np.diagonal(np.abs(block_k) @ np.abs(np.transpose(loo_coefs_j[:, coef_idx:(coef_idx + block_k.shape[1])])))
                elif self.lfi_abs == "outside":
                    lfi_matrix[:, k] = np.abs(np.diagonal(block_k @ np.transpose(loo_coefs_j[:, coef_idx:(coef_idx + block_k.shape[1])])))
                elif self.lfi_abs == "none":
                    lfi_matrix[:, k] = np.diagonal(block_k @ np.transpose(loo_coefs_j[:, coef_idx:(coef_idx + block_k.shape[1])]))
                else:
                    ValueError("lfi_abs must be either 'inside', 'outside', or 'none'.")
                coef_idx += block_k.shape[1]
            # print("equal:", np.allclose(lfi_matrix, lfi2_matrix))
        self.lfi_matrix = lfi_matrix
        self.n_features = blocked_data.n_blocks
        train_blocked_data, test_blocked_data, y_train, y_test, test_indices = \
            _get_sample_split_data(blocked_data, y, self.tree_random_state, self.sample_split)
        if train_blocked_data.get_all_data().shape[1] != 0:
            if hasattr(self.estimator, "predict_full") and \
                    hasattr(self.estimator, "predict_partial"):
                # print("IN IF STATEMENT IN LINE 389")
                full_preds = self.estimator.predict_full(test_blocked_data)
                # print("IN STATEMENT LINE 391")
                partial_preds = self.estimator.predict_partial(test_blocked_data, mode=self.mode, zero_values=zero_values)
            else:
                if self.task == "regression":
                    ppm = GenericRegressorPPM(self.estimator)
                elif self.task == "classification":
                    ppm = GenericClassifierPPM(self.estimator)
                # print("GETTING FULL PREDICTIONS")
                full_preds = ppm.predict_full(test_blocked_data)
                # print("GETTING PARTIAL PREDICTIONS")
                partial_preds = ppm.predict_partial(test_blocked_data, mode=self.mode)
            self._score_full_predictions(y_test, full_preds)
            self._score_partial_predictions(y_test, full_preds, partial_preds)

            full_preds_n = np.empty(n_samples) if full_preds.ndim == 1 \
                else np.empty((n_samples, full_preds.shape[1]))
            full_preds_n[:] = np.nan
            full_preds_n[test_indices] = full_preds
            self._full_preds = full_preds_n
        self.is_fitted = True

    def _score_full_predictions(self, y_test, full_preds):
        # print("IN '_score_full_predictions' METHOD WITHIN THE TREE MDI PLUS OBJECT")
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"score": self.scoring_fns}
        all_prediction_scores = pd.DataFrame({})
        for fn_name, scoring_fn in scoring_fns.items():
            scores = scoring_fn(y_test, full_preds)
            all_prediction_scores[fn_name] = [scores]
        self.prediction_score_ = all_prediction_scores

    def _score_partial_predictions(self, y_test, full_preds, partial_preds):
        # print("IN '_score_partial_predictions' METHOD WITHIN THE TREE MDI PLUS OBJECT")
        scoring_fns = self.scoring_fns if isinstance(self.scoring_fns, dict) \
            else {"importance": self.scoring_fns}
        if self.local_scoring_fns:
            if isinstance(self.local_scoring_fns, bool):
                local_scoring_fns = scoring_fns
            else:
                local_scoring_fns = self.local_scoring_fns if isinstance(self.local_scoring_fns, dict) \
                    else {"importance": self.local_scoring_fns}
        all_scores = pd.DataFrame({})
        all_local_scores = {}
        for fn_name, scoring_fn in scoring_fns.items():
            if self.local_scoring_fns:
                scores, local_scores = _partial_preds_to_scores(partial_preds, y_test, scoring_fn, local_scoring_fns[fn_name])
            else:
                scores = _partial_preds_to_scores(partial_preds, y_test, scoring_fn, self.local_scoring_fns)
            if self.mode == "keep_rest":
                full_score = scoring_fn(y_test, full_preds)
                scores = full_score - scores
            if len(partial_preds) != scores.size:
                if len(scoring_fns) > 1:
                    msg = "scoring_fn={} should return one value for each feature.".format(fn_name)
                else:
                    msg = "scoring_fns should return one value for each feature.".format(fn_name)
                raise ValueError("Unexpected dimensions. {}".format(msg))
            scores = scores.ravel()
            all_scores[fn_name] = scores
            if self.local_scoring_fns:
                all_local_scores[fn_name] = local_scores
        self.feature_importances_ = all_scores
        if self.local_scoring_fns:
            self.feature_importances_local_ = all_local_scores

def _partial_preds_to_scores(partial_preds, y_test, scoring_fn, local_scoring_fn=False):
    # print("IN '_partial_preds_to_scores' METHOD WITHIN THE TREE MDI PLUS OBJECT")
    scores = []
    local_scores = []
    for k, y_pred in partial_preds.items():
        if isinstance(y_pred, tuple):  # if constant model
            y_pred = np.ones_like(y_test) * y_pred[1]
        scores.append(scoring_fn(y_test, y_pred))
        if local_scoring_fn:
            if local_scoring_fn is True:
                local_scoring_fn = scoring_fn
            local_scores.append(
                [local_scoring_fn(y_test[i:(i+1), ], y_pred[i:(i+1), ]) for i in range(y_test.shape[0])]
            )
    if local_scoring_fn:
        return np.vstack(scores), np.vstack(local_scores).T
    else:
        return np.vstack(scores)

def _get_default_sample_split(sample_split, prediction_model, is_ppm):
    # print("IN '_get_default_sample_split' METHOD WITHIN THE TREE MDI PLUS OBJECT")
    if sample_split == "auto":
        sample_split = "oob"
        if is_ppm:
            if prediction_model.loo:
                sample_split = "loo"
    return sample_split


def _validate_sample_split(sample_split, prediction_model, is_ppm):
    # print("IN '_validate_sample_split' METHOD WITHIN THE TREE MDI PLUS OBJECT")
    if sample_split in ["oob", "inbag"] and is_ppm:
        if prediction_model.loo:
            raise ValueError("Cannot use LOO together with OOB or in-bag sample splitting.")


def _get_sample_split_data(blocked_data, y, random_state, sample_split):
    # print("IN '_get_sample_split_data' METHOD WITHIN THE TREE MDI PLUS OBJECT")
    if sample_split == "oob":
        train_blocked_data, test_blocked_data, y_train, y_test, _, test_indices = \
            _blocked_train_test_split(blocked_data, y, random_state)
    elif sample_split == "inbag":
        train_blocked_data, _, y_train, _, test_indices, _ = \
            _blocked_train_test_split(blocked_data, y, random_state)
        test_blocked_data = train_blocked_data
        y_test = y_train
    else:
        train_blocked_data = test_blocked_data = blocked_data
        y_train = y_test = y
        test_indices = np.arange(y.shape[0])
    return train_blocked_data, test_blocked_data, y_train, y_test, test_indices