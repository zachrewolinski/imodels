#General imports
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os
from collections import defaultdict 



# Lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader

#Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MetricCollection
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import BinaryF1Score, BinaryAveragePrecision, BinaryAUROC, BinaryAccuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


#sklearn imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_score, f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.base import BaseEstimator, RegressorMixin


#RF plus imports
import imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import _RandomForestPlus, RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier
from imodels.tree.rf_plus.rf_plus.MOE.moe_utils import TabularDataset, TreePlusExpert, AloTreePlusExpert, GatingNetwork

#Testing imports
import openml
import time

#
from xgboost import XGBRegressor


class RandomForestPlusMOE(pl.LightningModule):
    
    def __init__(self,rfplus_model, input_dim, noise_epsilon = 1e-2, criterion = nn.MSELoss(), gate_epsilon = 1e-10, use_loo = False,
                train_experts = False, lr = 1e-2, loss_coef = 0.01, noisy_gating = False,k = None, device_ = 'cpu',test_metrics = None):
        
        super(RandomForestPlusMOE,self).__init__()
       
        rfplus_model = copy.deepcopy(rfplus_model)  
        self.rfplus_model = rfplus_model
        self.task = rfplus_model._task
        self.input_dim = input_dim  
        self.noise_epsilon = noise_epsilon
        self.gate_epsilon = gate_epsilon
        self.loss_coef = loss_coef
        self.noisy_gating = noisy_gating
        self.transformers_ = [rfplus_model.transformers_[i] for i in range(len(rfplus_model.estimators_))]
        if use_loo:
            self.experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i].transformed_dim,train_experts) for i in range(len(rfplus_model.estimators_))])
        else:
            self.experts = nn.ModuleList([TreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i].transformed_dim,train_experts) for i in range(len(rfplus_model.estimators_))])
        self.num_experts = len(self.rfplus_model.estimators_)
        self.oob_indices_per_expert = [torch.tensor(rfplus_model._oob_indices[i])for i in range(self.num_experts)]
        self.gate = GatingNetwork(input_dim, self.num_experts,self.noise_epsilon,self.noisy_gating)
        self.lr = lr
        self.criterion = criterion
        self.train_experts = train_experts
        if k is None:
            self.k = self.num_experts
        else:
            self.k = k
            
        self.test_metrics = defaultdict()
        if test_metrics is None and self.task == "classification":
            self.prob_test_metrics = {"BinaryAUROC": BinaryAUROC(),"LogLoss": nn.BCELoss()} # "BinaryAveragePrecision": BinaryAveragePrecision()
            self.class_test_metrics = {"BinaryF1Score": BinaryF1Score(), "Accuracy": BinaryAccuracy()}
        elif test_metrics is None and self.task == "regression":
            self.test_metrics = {"R2Score": R2Score(), "MeanSquaredError": MeanSquaredError(), "MeanAbsoluteError": MeanAbsoluteError()}
        else:
            self.test_metrics = test_metrics
        
    def forward(self, x, index = None):
        gating_scores = self.gate(x)
        if self.training:
            batch_torch_indices = torch.tensor(index) #training indices of elements in batch
            all_oob_expert_indicator = [torch.isin(batch_torch_indices, expert_oob_indices) for expert_oob_indices in self.oob_indices_per_expert] #indicates which batch elements are oob for each expert
            all_oob_batch_indices = [torch.nonzero(oob_expert_indicator)for oob_expert_indicator in all_oob_expert_indicator] #indices of oob elements for each expert
            oob_mask = torch.zeros(x.shape[0],self.num_experts) 
            for i in range(self.num_experts):
                oob_mask[all_oob_batch_indices[i],i] = 1
            gating_scores = gating_scores * oob_mask 
        else:
            _ , topk_indices = gating_scores.topk(min(self.k, self.num_experts), dim=1, sorted=False) #shape (batch_size, k)
            mask = torch.zeros_like(gating_scores)
            mask.scatter_(1, topk_indices, 1)   
            gating_scores = gating_scores * mask
        
        gating_scores = F.softmax(gating_scores, dim=1)
            
        importance = gating_scores.sum(0)
        load = self._gates_to_load(gating_scores)
        loss = self._cv_squared(importance) + self._cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
        if self.task == "classification":
            expert_outputs = [torch.sigmoid(expert(self._apply_rfplus_transformer(x,self.transformers_[i]),index)) for i,expert in enumerate(self.experts)] #get probability of class 1 for each expert. Shape: batch_size x num_experts
        else:
            expert_outputs = [expert(self._apply_rfplus_transformer(x,self.transformers_[i]),index) for i,expert in enumerate(self.experts)] #get prediction for each expert. Shape: batch_size x num_experts
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)

        return expert_outputs, loss, gating_scores
    
    def _cv_squared(self, x):
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + self.gate_epsilon)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)       
        
    def _apply_rfplus_transformer(self,x,transformer_):
        x_numpy = x.clone().cpu().detach().numpy()
        transformed_x = transformer_.transform(x_numpy).get_all_data()
        return torch.tensor(transformed_x, dtype=torch.float32, device=x.device,requires_grad=True)
    
    def training_step(self,batch,batch_idx):
        data, target, index = batch
        output, aux_loss, _ = self(data, index)
        loss = self.criterion(output, target) + aux_loss
        self.log("train_loss", loss,on_step = True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        data, target, index = batch
        output, _, _ = self(data)
        loss = self.criterion(output, target)
        self.log("val_loss", loss,on_step = True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        data, target, index = batch
        output, _, _ = self(data)
        loss = self.criterion(output, target)

        if self.task == "classification":
                class_output = torch.round(output)
                for name,metric in self.prob_test_metrics.items():
                    name = "test_" + name
                    self.log("test_" + name, metric(output,target))
                for name,metric in self.class_test_metrics.items():
                    self.log("test_" +name, metric(class_output,target.long()),prog_bar=True)
        else:
            for name,metric in self.test_metrics.items():
                    self.log("test_" + name, metric(output,target),prog_bar=True)


        return loss
    
   
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['rfplus_model'] = self.rfplus_model
    #     checkpoint['input_dim'] = self.input_dim
    #     return checkpoint

    # def on_load_checkpoint(self, checkpoint):
    #     self.rfplus_model = checkpoint['rfplus_model']
    #     self.input_dim = checkpoint['input_dim']

class SklearnRFPlusRegMOE(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper for RandomForestPlusMOE Regressor."""
    
    def __init__(self, max_epochs=50, lr=1e-2, use_loo=False, 
                 train_experts=False, k=None, noise_epsilon=1e-2, gate_epsilon=1e-10,
                 loss_coef=0.01, noisy_gating=False, random_state=None, rfplus_model=None):
        self.max_epochs = max_epochs
        self.lr = lr
        self.use_loo = use_loo
        self.train_experts = train_experts
        self.k = k
        self.noise_epsilon = noise_epsilon
        self.gate_epsilon = gate_epsilon
        self.loss_coef = loss_coef
        self.noisy_gating = noisy_gating
        self.random_state = random_state
        self.rfplus_model = rfplus_model
        
    def fit(self, X, y):
        seed_everything(self.random_state)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state
        )
        
        
        # Prepare training data
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TabularDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=len(X_train))
        
        # Prepare validation data
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        val_dataset = TabularDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val))
        
        # Initialize and train MOE model
        self.model = RandomForestPlusMOE(
            rfplus_model=self.rfplus_model,
            input_dim=X.shape[1],
            criterion=nn.MSELoss(),
            use_loo=self.use_loo,
            train_experts=self.train_experts,
            lr=self.lr,
            k=self.k,
            noise_epsilon=self.noise_epsilon,
            gate_epsilon=self.gate_epsilon,
            loss_coef=self.loss_coef,
            noisy_gating=self.noisy_gating,
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        
        trainer = Trainer(
            accelerator="cpu", 
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[checkpoint_callback]
        )
        
        trainer.fit(self.model, train_dataloader, val_dataloader)
        # Get validation loss of best model
        best_val_loss = checkpoint_callback.best_model_score.item()
        self._val_loss = best_val_loss
        # Load the best model
        self.model = RandomForestPlusMOE.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            rfplus_model=self.rfplus_model,
            input_dim=X.shape[1],
            criterion=nn.MSELoss(),
            use_loo=self.use_loo,
            train_experts=self.train_experts,
            lr=self.lr,
            k=self.k,
            noise_epsilon=self.noise_epsilon,
            gate_epsilon=self.gate_epsilon,
            loss_coef=self.loss_coef,
            noisy_gating=self.noisy_gating,
        )
        
        return self
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        
        self.model.eval()
        self.model.training = False
        with torch.no_grad():
            predictions, _, _ = self.model(X)
        return predictions.cpu().numpy()
    
    def get_weights(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        self.model.training = False
        with torch.no_grad():
            _, _, gating_scores = self.model(X)
        return gating_scores.cpu().numpy()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "use_loo": self.use_loo,
            "train_experts": self.train_experts,
            "k": self.k,
            "noise_epsilon": self.noise_epsilon,
            "gate_epsilon": self.gate_epsilon,
            "loss_coef": self.loss_coef,
            "noisy_gating": self.noisy_gating,
            "random_state": self.random_state,
            "rfplus_model": self.rfplus_model
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class SklearnRFPlusRegMOEGrid(SklearnRFPlusRegMOE):
    def __init__(self, param_grid):
        self.param_grid = param_grid
       

    def fit(self, X, y):
       
        best_val_loss = float('inf')
        best_model = None
        
        for params in ParameterGrid(self.param_grid):
            rfplus_moe = SklearnRFPlusRegMOE(**params)
            rfplus_moe.fit(X, y)
            
            # Get validation loss from the trained model
            val_loss = rfplus_moe._val_loss
            
            # Update best model if current validation loss is lower
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(rfplus_moe)
                best_params = params
        
        # Store best model
        self.best_rfplus_moe = best_model
        self._val_loss = best_val_loss
        self.best_params = best_params
        return self
    
    def predict(self, X):
        return self.best_rfplus_moe.predict(X)

if __name__ == "__main__":
    # Load Data 
    suite_id = 353
    benchmark_suite = openml.study.get_suite(suite_id)
    task_ids = benchmark_suite.tasks
    task_id = 361236  # regression task
    random_state = 0
    task = "regression"
    seed_everything(random_state, workers=True)
    print(f"Task ID: {task_id}")
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)


    


    # Split data into train, validation, and test sets
    max_train = 500
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = copy.deepcopy(X_train)[:max_train], copy.deepcopy(y_train)[:max_train]
 
    #fit RF plus model
    if task == "classification":
        n_estimators = 256
        min_samples_leaf = 2
        max_epochs = 50
        max_features = "sqrt"
    else:
        n_estimators = 256
        min_samples_leaf = 5
        max_epochs = 50
        max_features = 0.33

    rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    rfplus_model = RandomForestPlusRegressor(rf_model = rf_model,fit_on = "all")
    rfplus_model.fit(X_train,y_train,n_jobs=-1)

    sklearn_rfplus_moe = SklearnRFPlusRegMOE(rfplus_model=rfplus_model) #BinaryF1ScoreBinaryF1Score
    sklearn_rfplus_moe.fit(X_train,y_train)

    sklearn_rfplus_moe_grid = SklearnRFPlusRegMOEGrid(param_grid={'rfplus_model':[rfplus_model],'lr':[1e-2]})
    sklearn_rfplus_moe_grid.fit(X_train,y_train)



    if rfplus_model._task == "classification":
        class_metrics = [accuracy_score, f1_score, precision_score]
        prob_metrics = [log_loss,roc_auc_score]
        for m in class_metrics:
            print(m.__name__)
            print("RF model: ",m(y_test,rf_model.predict(X_test)))
            print("RF+ Model: ",m(y_test,rfplus_model.predict(X_test)))
            print("\n")
        for m in prob_metrics:
            print(m.__name__)
            print("RF model: ",m(y_test,rf_model.predict_proba(X_test)[:,1]))
            print("RF+ Model: ",m(y_test,rfplus_model.predict_proba(X_test)[:,1]))
            print("\n")
    else:
        metrics = [mean_absolute_error,mean_squared_error, r2_score]
        for m in metrics:
            print(m.__name__)
            print("RF: ",m(y_test,rf_model.predict(X_test)))
            print("RF+: ",m(y_test,rfplus_model.predict(X_test)))
            print("RF+ (MOE): ",m(y_test,sklearn_rfplus_moe.predict(X_test)))
            print("RF+ (MOE Grid): ",m(y_test,sklearn_rfplus_moe_grid.predict(X_test)))
            print("\n")
    
    # Weights for each expert
    weights = sklearn_rfplus_moe.get_weights(X_test)
    num_experts = []
    expert_indices = []
   
    for i in range(weights.shape[0]):  # Iterate through each sample
        row_sum = 0
        num_needed = 0
        sorted_indices = np.argsort(weights[i])[::-1]  # Sort indices by weight in descending order
        row_indices = []
        for idx in sorted_indices:
            row_sum += weights[i][idx]
            num_needed += 1
            row_indices.append(idx)
            if row_sum >= 0.95:
                break
        num_experts.append(num_needed)
        expert_indices.append(row_indices)
    print(num_experts)
    print(expert_indices)





    # # rows_to_plot = [10,11,12]
    # # # Create a figure and axes
    # # fig, ax = plt.subplots()
    # # gating_scores = gating_scores.detach().numpy()
    # # # Set the width of each bar
    # # bar_width = 0.8 / len(rows_to_plot)

    # # # Iterate over the selected rows and plot them as bar plots
    # # for i, row_idx in enumerate(rows_to_plot):
    # #     row_data = gating_scores[row_idx]
    # #     x = np.arange(gating_scores.shape[1]) + i * bar_width
    # #     ax.bar(x, row_data, width=bar_width, label=f'Row {row_idx}')

    # # # Set the x-tick labels and positions
    # # x_ticks = np.arange(gating_scores.shape[1]) + bar_width * (len(rows_to_plot) - 1) / 2
    # # ax.set_xticks(x_ticks)
    # # ax.set_xticklabels([f'{i}' for i in range(gating_scores.shape[1])])

    # # # Add a legend
    # # ax.legend()

    # # #plt.show()





