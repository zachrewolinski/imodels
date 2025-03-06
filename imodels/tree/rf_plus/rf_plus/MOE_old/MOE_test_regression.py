#General imports
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from tabulate import tabulate
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
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import BinaryF1Score, BinaryAveragePrecision, BinaryAUROC, BinaryAccuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


#sklearn imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, precision_score, f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score


#RF plus imports
import imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import _RandomForestPlus, RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier
from imodels.tree.rf_plus.rf_plus.MOE.moe_utils import TabularDataset, TreePlusExpert, AloTreePlusExpert, GatingNetwork, get_columns_to_exceed_threshold
from imodels.tree.rf_plus.rf_plus.MOE.rfplus_MOE import RandomForestPlusMOE

#Testing imports
import openml, time, warnings, wandb

warnings.filterwarnings("ignore")

# Define the hyperparameter search space
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'min': 1e-5, 'max': 1e-1, 'distribution': 'log_uniform'},
        'loss_coef': {'min': 1e-5, 'max': 1e-1, 'distribution': 'log_uniform'},
        'train_experts': [True, False]}}

def train(config,rfplus_model,input_dim):
    with wandb.init(config=config):
        # Get the hyperparameters
        config = wandb.config
        lr = config.lr
        loss_coef = config.loss_coef
        #train_experts = config.train_experts

        # Create an instance of your model with the hyperparameters and additional objects
        model =  RandomForestPlusMOE(rfplus_model=rfplus_model, input_dim=input_dim, criterion= nn.MSELoss(),lr = lr,train_experts= train_experts)
        # Set up PyTorch Lightning trainer with wandb logger, early stopping, and model checkpoint
        wandb_logger = WandbLogger()
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',filename='best_model',monitor='val_loss',mode='min',save_top_k=1,verbose=True)
        trainer = pl.Trainer(max_epochs=40,logger=wandb_logger,callbacks=[early_stop_callback, checkpoint_callback])
        # Train and evaluate the model
        trainer.fit(model, train_dataloader, val_dataloader)

        # Load the best model
        best_model_path = checkpoint_callback.best_model_path
        best_model = RandomForestPlusMOE.load_from_checkpoint(best_model_path,rfplus_model=rfplus_model, input_dim=input_dim, criterion= nn.MSELoss(),lr = lr, use_loo = True, train_experts= train_experts, loss_coef = loss_coef)
        return best_model


if __name__ == "__main__":


    
    #Load Data 
    task_id =  359946
    random_state = 42
    seed_everything(random_state, workers=True)
    task = openml.tasks.get_task(task_id)
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    max_train = 250
    max_reps = 1


    n_estimators = 248
    min_samples_leaf = 5
    max_epochs = 25
    max_features = 0.33

    for fold in range(n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold)
        X,y = task.get_X_and_y(dataset_format="dataframe")
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        X_train, y_train = X_train[:max_train], y_train[:max_train]
        X_train, y_train, X_test, y_test = X_train.to_numpy(dtype = float), y_train.to_numpy(dtype = float), X_test.to_numpy(dtype = float), y_test.to_numpy(dtype = float)
        X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(copy.deepcopy(X_train),copy.deepcopy(y_train), test_size=0.2)
        
        
        #Get datasets and dataloaders
        train_dataset = TabularDataset(torch.tensor(X_train_torch), torch.tensor(y_train_torch))
        train_dataloader = DataLoader(train_dataset, batch_size=X_train_torch.shape[0])
        
        val_dataset = TabularDataset(torch.tensor(X_val_torch), torch.tensor(y_val_torch))
        val_dataloader = DataLoader(val_dataset, batch_size=X_val_torch.shape[0])
        
        test_dataset = TabularDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=X_test.shape[0])

        

        rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,random_state=random_state)
        rf_model.fit(X_train, y_train)


        rfplus_model = RandomForestPlusRegressor(rf_model = rf_model,fit_on = "all")
        rfplus_model.fit(X_train,y_train,n_jobs=-1)

        #Define the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',filename='best_model',monitor='val_loss',mode='min',save_top_k=1,save_last=True,verbose=True)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
        

        # Initialize wandb
        wandb.init(project='RFPlus_MOE_test')

        # Define the sweep
        sweep_id = wandb.sweep(sweep_config, project='RFPlus_MOE_test')

        # Start the sweep
        wandb.agent(sweep_id, function=train, rfplus_model=rfplus_model, input_dim=X.shape[1])

        
        RFplus_MOE = RandomForestPlusMOE(rfplus_model=rfplus_model, input_dim=X.shape[1], criterion= nn.MSELoss(),lr = 1e-3, use_loo = True, train_experts=  False) #BinaryF1ScoreBinaryF1Score
        logger = TensorBoardLogger(f'RFMOE_task_{task_id}', name='RFMOE')
        trainer = Trainer(max_epochs=max_epochs,callbacks=[checkpoint_callback])
        trainer.fit(RFplus_MOE, train_dataloader, val_dataloader)
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_val_model = RandomForestPlusMOE.load_from_checkpoint(best_model_path, rfplus_model=rfplus_model, input_dim=X.shape[1])




        rfplus_test_preds = rfplus_model.predict(X_test)
        best_val_model.eval()
        print(torch.tensor(X_test))
        rfplus_moe_test_preds,_, gating_scores = best_val_model(torch.tensor(X_test).float())
        rfplus_moe_test_preds = rfplus_moe_test_preds.detach().numpy()
        rftest_preds = rf_model.predict(X_test)


        metrics = [mean_absolute_error,mean_squared_error, r2_score]
        inference_time_metrics = ["avg_experts","median_experts","std_experts"]
        scores = {"RF" : {m.__name__: m(y_test,rftest_preds) for m in metrics}, "RFPlus" : {m.__name__ :m(y_test,rfplus_test_preds) for m in metrics}, "RFPlusMOE" : {m.__name__ :m(y_test,rfplus_moe_test_preds) for m in metrics}}
        for model in scores:
            for metric in inference_time_metrics:
                scores[model][metric] = n_estimators
            scores[model]["std_experts"] = 0.0
        
        #scores = pd.DataFrame.from_dict(scores, orient='index')
        

        num_experts_used  = get_columns_to_exceed_threshold(gating_scores, threshold=0.99)
        scores[model]["avg_experts"],scores[model]["median_experts"], scores[model]["std_experts"] = torch.mean(num_experts_used).item(), torch.median(num_experts_used).item(), torch.std(num_experts_used).item()
        scores = pd.DataFrame.from_dict(scores, orient='index')
        print(scores.to_markdown())

        if fold >= max_reps:
            break







    test = trainer.test(dataloaders=test_dataloader)

    




    
    for m in metrics:
            print(m.__name__)
            print("RF model: ",m(y_test,rf_model.predict(X_test)))
            print("RF+ Model without MOE: ",m(y_test,rfplus_model.predict(X_test)))
            print("RF+ Model with MOE: ",m(y_test, RFplus_MOE_preds))
            print("\n")
    