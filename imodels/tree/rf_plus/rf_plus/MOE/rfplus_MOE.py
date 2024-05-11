

# PyTorch 
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import R2Score
from pytorch_lightning.callbacks import ModelCheckpoint


#General imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

#sklearn imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

#RF plus imports
import imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import _RandomForestPlus, RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv import AloGLM
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier
from imodels.tree.rf_plus.rf_plus.MOE.moe_utils import TabularDataset, TreePlusExpert, AloTreePlusExpert, GatingNetwork


#Ray-tune imports 
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler

#Teesting imports
import openml
import time

import torch
# torch.manual_seed(0)
# def train(train_dataloader, val_dataloader, device, optimizer, criterion, forward, config):
#     for epoch in range(config["max_epochs"]):
#         train_loss = 0.0
#         for batch_idx, (data, target, index) in enumerate(train_dataloader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output, aux_loss, _ = forward(data, index)
#             loss = criterion(output, target)
#             loss += aux_loss
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_dataloader)
#         val_loss = val(val_dataloader, device, criterion, forward)
#         tune.report(train_loss=train_loss, val_loss=val_loss, epoch=epoch)


# def val(val_loader, device, criterion, forward):
#     eval()
#     val_loss = 0
#     with torch.no_grad():
#         for data, target, _ in val_loader:
#             data, target = data.to(device), target.to(device)
#             output, _, _ = forward(data)
#             val_loss += criterion(output, target).item()
#     val_loss /= len(val_loader)
#     return val_loss


# def _get_dataloader(X, y, shuffle=True):
#     dataset = TabularDataset(torch.tensor(X), torch.tensor(y))
#     dataloader = DataLoader(dataset, X.shape[0], shuffle=shuffle)
#     return dataloader


# def tune_model(X, y, rfplus_model, num_experts, oob_indices_per_expert, gate, train_dataloader, val_dataloader, scheduler, config):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio)
#     rfplus_model.fit(X_train, y_train)
#     if isinstance(rfplus_model.estimators_[0], AloGLM):
#         experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i], rfplus_model.transformers_[i]) for i in range(num_experts)])
#     else:
#         experts = nn.ModuleList([TreePlusExpert(rfplus_model.estimators_[i], rfplus_model.transformers_[i]) for i in range(num_experts)])

#     train_dataloader = _get_dataloader(X_train, y_train, True)
#     val_dataloader = _get_dataloader(X_val, y_val, False)

#     result = tune.run(lambda cfg: train(cfg, train_dataloader, val_dataloader),
#                       resources_per_trial={"cpu": 4},
#                       config=config,
#                       num_samples=3,
#                       scheduler=scheduler)

#     best_trial = result.get_best_trial("val_loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final training loss: {:.4f}".format(best_trial.last_result["train_loss"]))
#     print("Best trial final validation loss: {:.4f}".format(best_trial.last_result["val_loss"]))



class RandomForestPlusMOE(pl.LightningModule):
    def __init__(self,rfplus_model, input_dim, noise_epsilon = 1e-2, val_ratio = 0.2, criterion = nn.MSELoss(), 
                gate_epsilon = 1e-10, lr = 1e-2, loss_coef = 0.01, noisy_gating = False,k = None):
        
        super(RandomForestPlusMOE,self).__init__()
       
        self.noise_epsilon = noise_epsilon
        self.gate_epsilon = gate_epsilon
        self.loss_coef = loss_coef
        self.noisy_gating = noisy_gating
        self.rfplus_model = rfplus_model #trained RF plus model
        self.transformers_ = [rfplus_model.transformers_[i] for i in range(len(rfplus_model.estimators_))]
        if isinstance(rfplus_model.estimators_[0], AloGLM):
            self.experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i].transformed_dim) for i in range(len(rfplus_model.estimators_))])
        else:
            self.experts = nn.ModuleList([TreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i].transformed_dim) for i in range(len(rfplus_model.estimators_))])
        self.num_experts = len(self.rfplus_model.estimators_)
        self.oob_indices_per_expert = [torch.tensor(rfplus_model._oob_indices[i])for i in range(self.num_experts)]
        self.gate = GatingNetwork(input_dim, self.num_experts,self.noise_epsilon,self.noisy_gating)
        self.lr = lr
        self.criterion = criterion
        self.val_ratio = 0.2
        if k is None:
            self.k = self.num_experts
        else:
            self.k = k

    def cv_squared(self, x):
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + self.gate_epsilon)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)       
        
    def forward(self, x, index = None):
        gating_scores = self.gate(x)
        if self.training:
            batch_torch_indices = torch.tensor(index) #training indices of elements in batch
            all_oob_expert_indicator = [torch.isin(batch_torch_indices, expert_oob_indices) for expert_oob_indices in self.oob_indices_per_expert] #indicates which batch elements are oob for each expert
            all_oob_batch_elements = [batch_torch_indices[oob_expert_indicator] for oob_expert_indicator in all_oob_expert_indicator] #oob elements for each expert
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
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
        expert_outputs = torch.stack([expert(self._apply_rfplus_transformer(x,self.transformers_[i]),index) for i,expert in enumerate(self.experts)], dim=1)
        expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)

        # # Compute expert outputs
        # expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
        # expert_outputs = torch.stack([expert(x,index) for i,expert in enumerate(self.experts)], dim=1)
        # expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)
        
        return expert_outputs, loss, gating_scores
    
    def _apply_rfplus_transformer(self,x,transformer_):
        y = x.clone().detach().numpy()
        y = transformer_.transform(y).get_all_data()
        return torch.tensor(y, dtype=torch.float32, device=x.device,requires_grad=True)
    
    def training_step(self,batch,batch_idx):
        data, target, index = batch
        output, aux_loss, _ = self(data, index)
        loss = self.criterion(output, target) + aux_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        data, target, index = batch
        output, _, _ = self(data)
        loss = self.criterion(output, target) 
        self.log("val_loss", loss,prog_bar=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        data, target, index = batch
        output, _, _ = self(data)
        loss = self.criterion(output, target) 
        self.log("test_loss", loss)
        self.log("test_r2", r2_score(target, output))
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         print(f"Name of parameter: {name}")
    #         if param.requires_grad:
    #             self.logger.experiment.add_scalar(f"grads/{name}", param.grad.norm(), self.global_step)

       
    

if __name__ == "__main__":

    

    #Load Data 
    suite_id = 353
    benchmark_suite = openml.study.get_suite(suite_id)
    task_ids = benchmark_suite.tasks
    task_id =  361237
    random_state = 8
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
    X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(copy.deepcopy(X_train),
    copy.deepcopy(y_train), test_size=0.2)
    #Get datasets and dataloaders
    train_dataset = TabularDataset(torch.tensor(X_train_torch), torch.tensor(y_train_torch))
    train_dataloader = DataLoader(train_dataset, batch_size=X_train_torch.shape[0])
    
    val_dataset = TabularDataset(torch.tensor(X_val_torch), torch.tensor(y_val_torch))
    val_dataloader = DataLoader(val_dataset, batch_size=X_val_torch.shape[0])
    
    test_dataset = TabularDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=X_test.shape[0])

    #fit RF plus model
    n_estimators = 256
    min_samples_leaf = 5
    max_epochs = 10

    rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33)
    rf_model.fit(X_train, y_train)


    rfplus_model = RandomForestPlusRegressor(rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33), 
                                             prediction_model= AloElasticNetRegressorCV(l1_ratio=[0.0]),fit_on = "all")  
    rfplus_model.fit(X_train,y_train,n_jobs=-1)





    RFplus_MOEmodel = RandomForestPlusRegressor(rf_model=RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,max_features = 0.33),
                                                 prediction_model= AloElasticNetRegressorCV(l1_ratio=[0.0]),fit_on = "all")  
    RFplus_MOEmodel.fit(X_train,y_train,n_jobs=-1)

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',filename='best_model',monitor='val_loss',
    mode='min',save_top_k=1,save_last=True,verbose=True)
    
    
    RFplus_MOE = RandomForestPlusMOE(rfplus_model=RFplus_MOEmodel, input_dim=X.shape[1])
    logger = TensorBoardLogger(f'RFMOE_task_{task_id}', name='RFMOE')
    trainer = Trainer(max_epochs=max_epochs,callbacks=[checkpoint_callback],logger=logger)
    trainer.fit(RFplus_MOE, train_dataloader, val_dataloader)
    test = trainer.test(dataloaders=test_dataloader)

    # best_model_path = 'checkpoints/best_model.ckpt'
    # RFplus_MOE_best = RFplus_MOE.load_from_checkpoint(best_model_path, rfplus_model = copy.deepcopy(RFplus_MOEmodel), input_dim = X.shape[1])
    # RFplus_MOE_best.eval()
    # RFplus_MOE_preds = RFplus_MOE_best(torch.tensor(X_test,dtype=torch.float32),index = None)
    # RFplus_MOE_preds,_,gating_scores = RFplus_MOE_preds
    # RFplus_MOE_preds = RFplus_MOE_preds.detach().numpy()
    # gating_scores = gating_scores.detach().numpy()
   



    metrics = [mean_squared_error, r2_score]
    for m in metrics:
        print(m.__name__)
        print("RF model: ",m(y_test,rf_model.predict(X_test)))
        print("RF+ Model without MOE: ",m(y_test,rfplus_model.predict(X_test)))
        #print("RF+ Model with MOE: ",m(y_test,RFplus_MOE_preds))
        print("\n")
    


    


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







