# generic imports
import copy
from abc import ABC
import numpy as np
from scipy.special import expit
import time
from tqdm import tqdm
from joblib import Parallel, delayed

# sklearn imports
from sklearn.model_selection import train_test_split

# imodels/rfplus imports
import imodels 
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloGLMRegressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM

class MDIPlusGenericRegressorPPM(ABC):
    """
    Partial prediction model for arbitrary estimators. May be slow.
    """

    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        return self.estimator.predict(blocked_data.get_all_data())
    
    def predict_partial(self, blocked_data, mode, l2norm):
        """
        Gets the partial predictions. To be used when we want to incorporate the
        intercept of the regression model into the partial predictions.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k(blocked_data, k, mode,
                                                      l2norm)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, mode, l2norm, sign, normalize, njobs = 1):
        """
        Gets the partial predictions. To be used when we do not want to consider
        the intercept of the regression model, such as the partial linear LMDI+
        implementation.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sign (bool): indicator for if we want to retain the direction of
                         the partial prediction.
            normalize (bool): indicator for if we want to normalize the partial
                              predictions by the size of the full prediction.
            njobs (int): number of jobs to run in parallel.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        
        # helper function to parallelize the partial prediction computation
        def predict_wrapper(k):
            start_pred_k = time.time()
            predict_k = self.predict_partial_k_subtract_intercept(blocked_data,
                                                                  k, mode,
                                                                  l2norm, sign,
                                                                  normalize)
            end_time_k = time.time()
            return predict_k, end_time_k - start_pred_k
        
        start_partial_preds = time.time()
        # delayed makes sure that predictions get arranged in the correct order
        partial_preds = Parallel(n_jobs=njobs)(delayed(predict_wrapper)(k)
                                               for k in range(n_blocks))
        end_partial_preds = time.time()
        
        # parse through the outputs of the parallel data structure
        partial_pred_storage = {}
        pred_times = []
        for k in range(len(partial_preds)):
            partial_pred_storage[k] = partial_preds[k][0]
            pred_times.append(partial_preds[k][1])
            
        # save runtimes for analysis (can delete later if not needed)
        self._partial_preds_time = np.array(pred_times)
        self._total_partial_preds_time = end_partial_preds - start_partial_preds
        
        return partial_pred_storage
    
    def predict_partial_k(self, blocked_data, k, mode, l2norm):
        """
        Gets the partial predictions for an individual feature k, including the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            coefs = self.estimator.coef_
            # we square both the modified data and the coefficients to ensure
            # that there is no cancellation going on between positives and
            # negatives in the matrix multiplication - be careful with this,
            # as (AB)^2 != A^2B^2, so we need to do this consistently.
            return ((modified_data**2) @ (coefs**2)) ** (1/2) + \
                self.estimator.intercept_
        return self.estimator.predict(modified_data)

    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode,
                                             l2norm, sign, normalize):
        """
        Gets the partial predictions for an individual feature k, omitting the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        modified_data = blocked_data.get_modified_data(k, "only_k")
        if (sign or normalize) and (not l2norm):
            print("Warning: sign and normalize only work with l2norm=True.")
        if l2norm:
            # check if self.estimator has been fit
            if not hasattr(self.estimator, 'coef_'):
                print("Estimator has not been fit.")
            coefs = self.estimator.coef_
            # define sign_term and size to be 1 so that we can multiply/divide
            # by them even if we do not define them in the conditionals.
            sign_term = 1
            size = 1
            if sign:
                sign_term = np.sign(modified_data @ coefs)
            if normalize:
                all_data = blocked_data.get_all_data()
                size = ((all_data**2) @ (coefs**2)) ** (1/2)
            return sign_term * ((((modified_data**2) @ (coefs**2))**(1/2))/size)
        return modified_data @ self.estimator.coef_
        # return self.estimator.predict(modified_data) - self.estimator.intercept_

class MDIPlusGenericClassifierPPM(ABC):
    """
    Partial prediction model for arbitrary classification estimators. May be slow.
    """
    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        return self.estimator.predict_proba(blocked_data.get_all_data())[:,1]
    
    def predict_partial(self, blocked_data, mode, l2norm, sigmoid):
        """
        Gets the partial predictions. To be used when we want to incorporate the
        intercept of the regression model into the partial predictions.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k(blocked_data, k, mode,
                                                      l2norm, sigmoid)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, mode, l2norm,
                                           sign, sigmoid, normalize, njobs = 1):
        """
        Gets the partial predictions. To be used when we do not want to consider
        the intercept of the regression model, such as the partial linear LMDI+
        implementation.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sign (bool): indicator for if we want to retain the direction of
                         the partial prediction.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.
            normalize (bool): indicator for if we want to normalize the partial
                              predictions by the size of the full prediction.
            njobs (int): number of jobs to run in parallel.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        # helper function to parallelize the partial prediction computation
        def predict_wrapper(k):
            start_pred_k = time.time()
            predict_k = self.predict_partial_k_subtract_intercept(blocked_data,
                                                                  k, mode,
                                                                  l2norm,
                                                                  sigmoid,
                                                                  sign,
                                                                  normalize)
            end_time_k = time.time()
            return predict_k, end_time_k - start_pred_k
        
        start_partial_preds = time.time()
        # delayed makes sure that predictions get arranged in the correct order
        partial_preds = Parallel(n_jobs=njobs)(delayed(predict_wrapper)(k)
                                               for k in range(n_blocks))
        end_partial_preds = time.time()
        # parse through the outputs of the parallel data structure
        partial_pred_storage = {}
        pred_times = []
        for k in range(len(partial_preds)):
            partial_pred_storage[k] = partial_preds[k][0]
            pred_times.append(partial_preds[k][1])
            
        # save runtimes for analysis (can delete later if not needed)
        self._partial_preds_time = np.array(pred_times)
        self._total_partial_preds_time = end_partial_preds - start_partial_preds
        
        return partial_pred_storage
    
    def predict_partial_k(self, blocked_data, k, mode, l2norm, sigmoid):
        """
        Gets the partial predictions for an individual feature k, including the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        modified_data = blocked_data.get_modified_data(k, mode)
        coefs = self.estimator.coef_
        if l2norm:
            ppred = ((modified_data**2) @ (coefs**2))**(1/2) + \
                self.estimator.intercept_
            if sigmoid:
                return expit(ppred)
            else:
                return ppred
        if sigmoid:
            return self.estimator.predict_proba(modified_data)[:,1]
        return modified_data @ coefs + self.estimator.intercept_
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k, mode,
                                             l2norm, sigmoid, sign, normalize):
        """
        Gets the partial predictions for an individual feature k, omitting the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.
            sign (bool): indicator for if we want to retain the direction of
                            the partial prediction.
            normalize (bool): indicator for if we want to normalize the partial
                                predictions by the size of the full prediction.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        modified_data = blocked_data.get_modified_data(k, "only_k")
        coefs = self.estimator.coef_
        # reshape coefs if necessary
        if coefs.shape[0] != modified_data.shape[1]:
            coefs = coefs.reshape(-1,1)
        # print("Prediction:")
        # print(modified_data @ coefs + self.estimator.intercept_)
        sign_term = 1
        size = 1
        if l2norm:
            if sigmoid:
                return expit(((modified_data**2) @ (coefs**2)))
            if sign:
                sign_term = np.sign(modified_data @ coefs)
            if normalize:
                all_data = blocked_data.get_all_data()
                size = ((all_data**2) @ (coefs**2))
            return sign_term * (((modified_data**2) @ (coefs**2)))/size
        if sigmoid:
            return  expit(modified_data @ coefs)
        return modified_data @ coefs


class AloMDIPlusPartialPredictionModelRegressor(MDIPlusGenericRegressorPPM,AloGLMRegressor):
    """
    Assumes that the estimator has a loo_coefficients_ attribute.
    """

    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_loo(blocked_data.get_all_data())

    def predict_partial_loo(self, blocked_data, mode, l2norm):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode, l2norm)
        return partial_preds

    def predict_partial_loo_subtract_intercept(self, blocked_data, mode, l2norm, sign):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode, l2norm, sign=sign)
        return partial_preds

    def predict_partial_k_loo(self, blocked_data, k, mode, l2norm):
        modified_data = blocked_data.get_modified_data(k, mode)
        if l2norm:
            coefs = self.estimator.loo_coefficients_[:, :-1] 
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1]
        return self.estimator.predict_loo(modified_data)
   
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode, l2norm, sign):
        modified_data = blocked_data.get_modified_data(k, "only_k")
        if l2norm:
            coefs = self.estimator.loo_coefficients_[:, :-1]
            if sign:
                signs = np.sign(np.sum(modified_data * coefs, axis=1))
                return signs*np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
        return np.sum(modified_data * coefs, axis=1)
        #return self.estimator.predict_loo(modified_data) - self.estimator.loo_coefficients_[:, -1]
    

class AloMDIPlusPartialPredictionModelClassifier(MDIPlusGenericClassifierPPM,AloGLMClassifier):
    """
    Assumes that the estimator has a loo_coefficients_ attribute.
    """    
    def predict_full_loo(self, blocked_data):
        return self.estimator.predict_proba_loo(blocked_data.get_all_data())[:,1]

    def predict_partial_loo(self, blocked_data, mode, l2norm, sigmoid):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo(blocked_data, k, mode, l2norm, sigmoid)
        return partial_preds
    
    def predict_partial_loo_subtract_intercept(self, blocked_data, mode, l2norm, sigmoid, sign):
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k_loo_subtract_intercept(blocked_data, k, mode, l2norm, sigmoid, sign=sign)
        return partial_preds

    def predict_partial_k_loo(self, blocked_data, k, mode, l2norm, sigmoid):
        modified_data = blocked_data.get_modified_data(k, mode)
        coefs = self.estimator.loo_coefficients_[:, :-1]
        if l2norm:
            if sigmoid:
                return expit(np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1])
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1) + self.estimator.loo_coefficients_[:, -1]
        if sigmoid:
            return self.estimator.predict_proba_loo(modified_data)[:,1]
        return np.sum((modified_data * coefs), axis = 1) + self.estimator.loo_coefficients_[:, -1]
    
    def predict_partial_k_loo_subtract_intercept(self, blocked_data, k, mode, l2norm, sigmoid, sign):
        modified_data = blocked_data.get_modified_data(k, "only_k")
        coefs = self.estimator.loo_coefficients_[:, :-1]
        if l2norm:
            if sigmoid:
                return expit(np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1))
            if sign:
                signs = np.sign(np.sum(modified_data * coefs, axis=1))
                return signs*np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
            return np.sum(((modified_data**2) * (coefs**2))**(1/2), axis = 1)
        if sigmoid:
            return expit(np.sum(modified_data * coefs, axis = 1))
        return np.sum(modified_data * coefs, axis = 1)

if __name__ == "__main__":
    
    # Test MDIPlus Regression
    X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    rfplus_reg = RandomForestPlusRegressor()
    rfplus_reg.fit(X_train, y_train)

    for i in tqdm(range(len(rfplus_reg.estimators_))):
        mdiplus_ppm = AloMDIPlusPartialPredictionModelRegressor(rfplus_reg.estimators_[i])
        train_blocked_data_i = rfplus_reg.transformers_[i].transform(X_train)
        test_blocked_data_i = rfplus_reg.transformers_[i].transform(X_test)
        partial_loo_preds = mdiplus_ppm.predict_partial_loo(train_blocked_data_i,mode="keep_k",l2norm=False)
        partial_preds = mdiplus_ppm.predict_partial(test_blocked_data_i,mode="keep_k", l2norm=False)
        print(f"Partial preds number of features: {len(partial_preds)}")
        print(f"Partial preds number of samples per feature: {len(partial_preds[0])}")

        print(f"LOO Partial preds number of features: {len(partial_loo_preds)}")
        print(f"LOO Partial preds number of samples per feature: {len(partial_loo_preds[0])}")
        break
    

    # Test MDIPlus Classification
    # X,y,f = imodels.get_clean_dataset("enhancer",data_source="imodels")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    # rfplus_clf = RandomForestPlusClassifier()
    # rfplus_clf.fit(X_train, y_train)

    # print(X_train.shape)

    # print("Done fitting rfplus classifier")
   
    # for i in range(len(rfplus_clf.estimators_)):
    #     mdiplus_ppm_clf = AloMDIPlusPartialPredictionModelRegressor(rfplus_clf.estimators_[i])
    #     train_blocked_data_i = rfplus_clf.transformers_[i].transform(X_train)
    #     test_blocked_data_i = rfplus_clf.transformers_[i].transform(X_test)
    #     partial_loo_preds = mdiplus_ppm_clf.predict_partial_loo(train_blocked_data_i,mode="keep_k")
    #     partial_preds = mdiplus_ppm_clf.predict_partial(test_blocked_data_i,mode="keep_k")
    #     print(f"Partial preds: {partial_preds}")
    #     break








    
    
    
    
    
    
    # def _fit_model(self, X, y):
    #     self.estimator.fit(X, y)

    # def predict(self, X):
    #     return self.estimator.predict(X)
    
    # @abstractmethod
    # def predict(self, X):
    #     """
    #     Make predictions on new data using the fitted model.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix, for which to make predictions.
    #     """
    #     pass

    #self.is_fitted = False

    # def fit(self, X, y):
    #     """
    #     Fit the partial prediction model.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix.
    #     y: ndarray of shape (n_samples, n_targets)
    #         The observed responses.
    #     """
    #     self._fit_model(X, y)
    #     self.is_fitted = True

    # @abstractmethod
    # def _fit_model(self, X, y):
    #     """
    #     Fit the regression or classification model on all the data.

    #     Parameters
    #     ----------
    #     X: ndarray of shape (n_samples, n_features)
    #         The covariate matrix.
    #     y: ndarray of shape (n_samples, n_targets)
    #         The observed responses.
    #     """
    #    pass