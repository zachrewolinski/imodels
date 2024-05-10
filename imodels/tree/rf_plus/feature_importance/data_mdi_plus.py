     
    # def get_kernel_shap_scores(self, X_train: np.ndarray, X_test: np.ndarray,
    #                            p: float = 1, use_summary: bool = True,
    #                            k : int = 1) -> np.ndarray:
    #     """
    #     Obtain KernelSHAP feature importances.

    #     Inputs:
    #         X_train (np.ndarray): The training covariate matrix. This is
    #                              necessary to fit the KernelSHAP model.
    #         X_test (np.ndarray): The testing covariate matrix. This is the data
    #                             the resulting SHAP values will be based on.
    #         p (float): The proportion of the training data which will be used to
    #                    fit the KernelSHAP model. Due to the expensive
    #                    computation of KernelSHAP, for large datasets it may be
    #                    helpful to have p < 1.
    #         use_summary (bool): Whether to use the summary of the SHAP values
    #                             via shap.kmeans
    #         k (int): The number of clusters to use for the shap.kmeans algorithm
    #     """
        
    #     if self._task == "regression": 
    #         model_pred_func = self.predict 
    #     else: 
    #         model_pred_func = self.predict_proba
    #     return _get_kernel_shap_rf_plus(model_pred_func,self._task,X_train,X_test,p,use_summary,k, self.random_state)

    # def get_lime_scores(self, X_train: np.ndarray,
    #                     X_test: np.ndarray) -> np.ndarray:
    #     """
    #     Obtain LIME feature importances.

    #     Inputs:
    #         X_train (np.ndarray): The training covariate matrix. This is
    #                              necessary to fit the LIME model.
    #         X_test (np.ndarray): The testing covariate matrix. This is the data
    #                             the resulting LIME values will be based on.
    #         num_samples (int): The number of samples to use when fitting the
    #                            LIME model.
    #     """
    #     if self._task == "regression": 
    #         model_pred_func = self.predict
    #     else:
    #         model_pred_func = self.predict_proba
    #     return _get_lime_scores_rf_plus(model_pred_func,self._task,X_train,X_test,self.random_state)
    
    