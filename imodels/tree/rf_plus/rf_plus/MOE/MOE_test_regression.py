#General imports
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
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
from sklearn.model_selection import train_test_split, cross_val_score


#RF plus imports
import imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import _RandomForestPlus, RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier
from imodels.tree.rf_plus.rf_plus.MOE.moe_utils import TabularDataset, TreePlusExpert, AloTreePlusExpert, GatingNetwork
from imodels.tree.rf_plus.rf_plus.MOE.rfplus_MOE import RandomForestPlusMOE

#Testing imports
import openml
import time




if __name__ == "__main__":

    
    #Load Data 
    suite_id = 353
    benchmark_suite = openml.study.get_suite(suite_id)
    task_ids = benchmark_suite.tasks
    task_id =  359945
    random_state = 42
    seed_everything(random_state, workers=True)
    print(f"Task ID: {task_id}")
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)


    


    # Split data into train, validation, and test sets
    max_train = 1000
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = copy.deepcopy(X_train)[:max_train], copy.deepcopy(y_train)[:max_train]
    X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(copy.deepcopy(X_train),copy.deepcopy(y_train), test_size=0.2)
    
    
    #Get datasets and dataloaders
    train_dataset = TabularDataset(torch.tensor(X_train_torch), torch.tensor(y_train_torch))
    train_dataloader = DataLoader(train_dataset, batch_size=X_train_torch.shape[0])
    
    val_dataset = TabularDataset(torch.tensor(X_val_torch), torch.tensor(y_val_torch))
    val_dataloader = DataLoader(val_dataset, batch_size=X_val_torch.shape[0])
    
    test_dataset = TabularDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=X_test.shape[0])

    
    n_estimators = 256
    min_samples_leaf = 5
    max_epochs = 50
    max_features = 0.33

    rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,random_state=random_state)
    rf_model.fit(X_train, y_train)


    rfplus_model = RandomForestPlusRegressor(rf_model = rf_model,fit_on = "all")
    rfplus_model.fit(X_train,y_train,n_jobs=-1)

    #Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',filename='best_model',monitor='val_loss',mode='min',save_top_k=1,save_last=True,verbose=True)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    
    
    RFplus_MOE = RandomForestPlusMOE(rfplus_model=rfplus_model, input_dim=X.shape[1], criterion= nn.BCELoss(), use_loo = True, train_experts=  False) #BinaryF1ScoreBinaryF1Score
    logger = TensorBoardLogger(f'RFMOE_task_{task_id}', name='RFMOE')
    trainer = Trainer(max_epochs=max_epochs,callbacks=[checkpoint_callback])
    trainer.fit(RFplus_MOE, train_dataloader, val_dataloader)
    test = trainer.test(dataloaders=test_dataloader)

    rfplus_test_preds = rfplus_model.predict(X_test)
    rftest_preds = rf_model.predict(X_test)

    metrics = [mean_absolute_error,mean_squared_error, r2_score]
    scores = {"rfplus" : {m.__name__: m(y_test,rftest_preds) for m in metrics}, "rfplus_moe" : {m.__name__ :m(y_test,rfplus_test_preds) for m in metrics}}
    
    # for m in metrics:
    #         print(m.__name__)
    #         print("RF model: ",m(y_test,rf_model.predict(X_test)))
    #         print("RF+ Model without MOE: ",m(y_test,rfplus_model.predict(X_test)))
    #         #print("RF+ Model with MOE: ",m(y_test,RFplus_MOE_preds))
    #         print("\n")
    