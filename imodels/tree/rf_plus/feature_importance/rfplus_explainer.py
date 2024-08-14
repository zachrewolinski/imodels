# Generic imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from functools import partial
import copy
import pprint

# Sklearn imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LogisticRegressionCV

#imports from imodels
import imodels
from imodels.tree.rf_plus.feature_importance.ppms.ppms import MDIPlusGenericRegressorPPM, MDIPlusGenericClassifierPPM
from imodels.tree.rf_plus.feature_importance.ppms.ppms import AloMDIPlusPartialPredictionModelRegressor, AloMDIPlusPartialPredictionModelClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloGLMRegressor, AloElasticNetRegressorCV
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV
from imodels.tree.rf_plus.data_transformations.block_transformers import BlockTransformerBase, _blocked_train_test_split, BlockPartitionedData
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier

#Wandb
import wandb

#Feature importance methods
import shap,lime


def per_sample_neg_log_loss(y_true,y_pred,epsilon = 1e-4):

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = y_true.reshape(-1,1)
    log_loss_values = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return -1*log_loss_values

def per_sample_neg_mean_absolute_error(y_true,y_pred):
    y_true = y_true.reshape(-1,1)   
    return -1*np.abs(y_true - y_pred)

class _RandomForestPlusExplainer():

    def __init__(self, rf_plus_model):
        self.rf_plus_model = rf_plus_model
    
    def explain(self, X):
        pass 

    def get_rankings(self,scores,feature_names = None,ascending = True):
        """
        Takes in a matrix of scores and returns rankings of columns per row. 
        Ascending: higher scores are better. 
        """
        if ascending:
            rankings = np.argsort(-scores, axis=1)
        else:
            rankings = np.argsort(scores, axis=1)
        if feature_names is None:
            return rankings
        else:
            return pd.DataFrame(rankings, columns = feature_names)

class RFPlusKernelSHAP(_RandomForestPlusExplainer):

    def __init__(self, rf_plus_model):
        self.rf_plus_model = rf_plus_model
        self.task = rf_plus_model._task
        if self.task == "classification":
            self.model_pred_func = self.rf_plus_model.predict_proba
        else:
            self.model_pred_func = self.rf_plus_model.predict  
    
    def explain(self, X_train,X_test,p = 0.5,use_summary = True,k = 3,num_features = 10):
        # check that p is a valid proportion
        assert 0 < p <= 1, "p must be in the interval (0, 1]"

        # for faster computation, we may want to use shap.kmeans
        if use_summary:
            X_train = shap.kmeans(X_train, k)
        else:
            X_train = shap.utils.sample(X_train,int(p * X_train.shape[0]))
    
        # fit the KernelSHAP model and get the SHAP values
        ex = shap.KernelExplainer(self.model_pred_func, X_train)

        num_features = "num_features(" + str(num_features) + ")"

        if X_test is None: #assume we are explaining training set
            shap_values = ex.shap_values(X_train,l1_reg = num_features)
        else: #assume we are explaining test set
            shap_values = ex.shap_values(X_test,l1_reg = num_features)

        # if self.task == "classification":
        #     shap_values = np.sum(np.abs(shap_values),axis=-1)
        # else:
        #     shap_values = shap_values #abs(shap_values)
        return shap_values
    
class RFPlusLime(_RandomForestPlusExplainer):

    def __init__(self, rf_plus_model):
        self.rf_plus_model = rf_plus_model
        self.task = rf_plus_model._task
        if self.task == "classification":
            self.model_pred_func = self.rf_plus_model.predict_proba
        else:
            self.model_pred_func = self.rf_plus_model.predict  
    
    def explain(self, X_train,X_test,num_features = 10): #For experiments change based on number of features we are ablating 
        
        # get shape of X_test
        if X_test is None: #assume we are explaining training set
            X_to_explain = copy.deepcopy(X_train)   
            n_samples, num_features = X_train.shape
        else: #assume we are explaining test set
            X_to_explain = copy.deepcopy(X_test)
            n_samples, num_features = X_test.shape
        
        # create data structure to save scores in
        result = np.zeros((n_samples, num_features))
        
        # initialize the LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train,verbose=False,mode=self.task)
        
        for i in range(n_samples):
            exp = explainer.explain_instance(X_to_explain[i,:], self.model_pred_func,num_features=num_features)
            original_feature_importance = exp.as_map()[1]
            sorted_feature_importance = sorted(original_feature_importance,key = lambda x: x[0])
            for j in range(num_features):
                result[i,j] = sorted_feature_importance[j][1] #abs(sorted_feature_importance[j][1])
        
        # Convert the array to a DataFrame
        lime_values = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])
        lime_values = lime_values #abs(lime_values)
        
        return lime_values
        
class RFPlusMDI(_RandomForestPlusExplainer): #No leave one out 

    def __init__(self, rf_plus_model,mode = 'keep_k', evaluate_on = 'oob'):
        self.rf_plus_model = rf_plus_model
        self.mode = mode
        self.oob_indices = self.rf_plus_model._oob_indices
        self.evaluate_on = evaluate_on #training feature importances 

        if self.rf_plus_model._task == "classification":
            self.tree_explainers = [MDIPlusGenericClassifierPPM(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
            self.train_metrics = per_sample_neg_log_loss
            self.test_metrics = per_sample_neg_mean_absolute_error
        else:
            self.tree_explainers = [MDIPlusGenericRegressorPPM(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
            self.train_metrics = per_sample_neg_mean_absolute_error
            self.test_metrics = per_sample_neg_mean_absolute_error
        

    def explain(self, X,y = None):
        """
        If y is None, return the local feature importance scores for X. 
        If y is not None, assume X is FULL training set
        """
        local_feature_importances = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers))) #partial predictions for each sample  
        partial_preds = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        local_feature_importances[local_feature_importances == 0] = np.nan
        partial_preds[partial_preds == 0] = np.nan
     
        # all_tree_LFI_scores has shape X.shape[0], X.shape[1], num_trees 
        all_tree_LFI_scores,all_tree_full_preds = self._get_LFI(X,y)
        
        if y is None:
            y = all_tree_full_preds
            evaluate_on = None
            metric = self.test_metrics
        else:
            y = np.hstack([y.reshape(-1,1) for _ in range(len(self.tree_explainers))])
            evaluate_on = self.evaluate_on
            metric = self.train_metrics
        
        for i in range(all_tree_full_preds.shape[1]):
            ith_partial_preds = all_tree_LFI_scores[:,:,i]
            ith_tree_scores = metric(y[:,i],ith_partial_preds)
            if evaluate_on == 'oob':
                oob_indices = np.unique(self.oob_indices[i])
                local_feature_importances[oob_indices,:,i] = ith_tree_scores[oob_indices,:]
                partial_preds[oob_indices,:,i] = ith_partial_preds[oob_indices,:]
            elif evaluate_on == 'inbag':
                oob_indices = np.unique(self.oob_indices[i])
                inbag_indices = np.arange(X.shape[0])
                inbag_indices = np.setdiff1d(inbag_indices,oob_indices)
                local_feature_importances[inbag_indices,:,i] = ith_tree_scores[inbag_indices,:]
                partial_preds[inbag_indices,:,i] = ith_partial_preds[inbag_indices,:]
            else:
                local_feature_importances[:,:,i] = ith_tree_scores
                partial_preds[:,:,i] = ith_partial_preds
        
        local_feature_importances = np.nanmean(local_feature_importances,axis=-1)
        partial_preds = np.nanmean(partial_preds,axis=-1)
        return local_feature_importances, partial_preds
    
    def explain_subtract_intercept(self, X,y = None):
        """
        If y is None, return the local feature importance scores for X. 
        If y is not None, assume X is FULL training set
        """
        local_feature_importances = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers))) #partial predictions for each sample  
        local_feature_importances[local_feature_importances == 0] = np.nan
     
        # all_tree_LFI_scores has shape X.shape[0], X.shape[1], num_trees 
        all_tree_LFI_scores = self._get_LFI_subtract_intercept(X,y)
        
        if y is None:
            evaluate_on = None
        else:
            evaluate_on = self.evaluate_on
        
        for i in range(all_tree_LFI_scores.shape[-1]):
            ith_partial_preds = all_tree_LFI_scores[:,:,i]
            ith_tree_scores = ith_partial_preds
            if evaluate_on == 'oob':
                oob_indices = np.unique(self.oob_indices[i])
                local_feature_importances[oob_indices,:,i] = ith_tree_scores[oob_indices,:]
            elif evaluate_on == 'inbag':
                oob_indices = np.unique(self.oob_indices[i])
                inbag_indices = np.arange(X.shape[0])
                inbag_indices = np.setdiff1d(inbag_indices,oob_indices)
                local_feature_importances[inbag_indices,:,i] = ith_tree_scores[inbag_indices,:]
            else:
                local_feature_importances[:,:,i] = ith_tree_scores
        
        local_feature_importances = np.nanmean(local_feature_importances,axis=-1)
        return local_feature_importances
              
    def _get_LFI(self, X,y):
        LFIs = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        full_preds = np.zeros((X.shape[0],len(self.tree_explainers)))
        for i, tree_explainer in enumerate(self.tree_explainers):
            blocked_data_ith_tree = self.rf_plus_model.transformers_[i].transform(X)
            ith_partial_preds = tree_explainer.predict_partial(blocked_data_ith_tree, mode=self.mode)
            ith_partial_preds = np.array([ith_partial_preds[j] for j in range(X.shape[1])]).T
            LFIs[:,:,i] = ith_partial_preds
            full_preds[:,i] = tree_explainer.predict_full(blocked_data_ith_tree)
        return LFIs, full_preds
    
    def _get_LFI_subtract_intercept(self, X,y):
        LFIs = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        for i, tree_explainer in enumerate(self.tree_explainers):
            blocked_data_ith_tree = self.rf_plus_model.transformers_[i].transform(X)
            ith_partial_preds = tree_explainer.predict_partial_subtract_intercept(blocked_data_ith_tree, mode=self.mode)
            ith_partial_preds = np.array([ith_partial_preds[j] for j in range(X.shape[1])]).T
            LFIs[:,:,i] = ith_partial_preds
        return LFIs

       
class AloRFPlusMDI(RFPlusMDI): #Leave one out 

    def __init__(self, rf_plus_model,mode = 'keep_k', evaluate_on = 'oob'):
        self.rf_plus_model = rf_plus_model
        self.mode = mode
        self.oob_indices = self.rf_plus_model._oob_indices
        self.evaluate_on = evaluate_on #training feature importances 

        if self.rf_plus_model._task == "classification":
            self.tree_explainers = [AloMDIPlusPartialPredictionModelClassifier(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
            self.train_metrics = per_sample_neg_log_loss
            self.test_metrics = per_sample_neg_mean_absolute_error
        else:
            self.tree_explainers = [AloMDIPlusPartialPredictionModelRegressor(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
            self.train_metrics = per_sample_neg_mean_absolute_error
            self.test_metrics = per_sample_neg_mean_absolute_error
        
    def explain(self, X,y = None):
        return super().explain(X,y)
    
    def explain_subtract_intercept(self, X,y = None):
        return super().explain_subtract_intercept(X,y)

    def _get_LFI(self,X,y):
        LFIs = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        full_preds = np.zeros((X.shape[0],len(self.tree_explainers)))
        for i, tree_explainer in enumerate(self.tree_explainers):
            blocked_data_ith_tree = self.rf_plus_model.transformers_[i].transform(X)
            if y is None:
                ith_partial_preds = tree_explainer.predict_partial(blocked_data_ith_tree, mode=self.mode)
                full_preds[:,i] = tree_explainer.predict_full(blocked_data_ith_tree)
            else:
                ith_partial_preds = tree_explainer.predict_partial_loo(blocked_data_ith_tree, mode=self.mode)
                full_preds[:,i] = tree_explainer.predict_full_loo(blocked_data_ith_tree)
            ith_partial_preds = np.array([ith_partial_preds[j] for j in range(X.shape[1])]).T
            LFIs[:,:,i] = ith_partial_preds

        return LFIs, full_preds

    def _get_LFI_subtract_intercept(self,X,y):
        LFIs = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        for i, tree_explainer in enumerate(self.tree_explainers):
            blocked_data_ith_tree = self.rf_plus_model.transformers_[i].transform(X)
            if y is None:
                ith_partial_preds = tree_explainer.predict_partial_subtract_intercept(blocked_data_ith_tree, mode=self.mode)
            else:
                ith_partial_preds = tree_explainer.predict_partial_loo_subtract_intercept(blocked_data_ith_tree, mode=self.mode)
            ith_partial_preds = np.array([ith_partial_preds[j] for j in range(X.shape[1])]).T
            LFIs[:,:,i] = ith_partial_preds
        return LFIs



            
            
        
if __name__ == "__main__":

    X, y, f = imodels.get_clean_dataset("diabetes")
    pprint.pprint(f"X Shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Fit a RFPlus model
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,max_features='sqrt', random_state=42)
    
    rf_plus_model = RandomForestPlusClassifier(rf_model = rf_model)
    rf_plus_model.fit(X_train[:100], y_train[:100])

    #Test MDI
    rf_plus_mdi = AloRFPlusMDI(rf_plus_model, evaluate_on="all")
    local_feature_importances, partial_preds = rf_plus_mdi.explain(X_test[:40])
    pprint.pprint(rf_plus_mdi.get_rankings(local_feature_importances,f))

    rf_plus_model = RandomForestPlusClassifier(rf_model = rf_model, fit_on="oob")
    rf_plus_model.fit(X_train[:100], y_train[:100])


    rf_plus_mdi = AloRFPlusMDI(rf_plus_model, evaluate_on="all")
    local_feature_importances, partial_preds = rf_plus_mdi.explain(X_test[:40])
    pprint.pprint(rf_plus_mdi.get_rankings(local_feature_importances,f))

