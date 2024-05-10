

# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

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
torch.manual_seed(0)


class _RandomForestPlusRMOE(nn.Module):
    def __init__(self,rf_model=None, prediction_model=None, include_raw=True, drop_features=True, add_transformers=None, 
                 center=True, normalize=False, fit_on="all", verbose=True, warm_start=False, lr = 1e-2, val_ratio = 0.2, optimizer = torch.optim.Adam,
                    criterion = nn.MSELoss(), max_epochs = 10, noise_epsilon = 1e-2, gate_epsilon = 1e-10, loss_coef = 0.01, noisy_gating = False,):
        
        rfplus_model = RandomForestPlusRegressor(rf_model=rf_model, prediction_model=prediction_model, include_raw=include_raw, drop_features=drop_features, 
                                                add_transformers=add_transformers, center=center, normalize=normalize, fit_on=fit_on,verbose=verbose, 
                                                warm_start=warm_start)
        self.rfplus_model = rfplus_model
        self.num_experts = len(rfplus_model.estimators_)
        self.lr = lr
        self.val_ratio = val_ratio
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.noise_epsilon = noise_epsilon
        self.loss_coef = loss_coef
        self.noisy_gating = noisy_gating
        self.gate_epsilon = gate_epsilon
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {"max_epochs": self.max_epochs, "lr": tune.loguniform(1e-4, 1e-1), "loss_coef": tune.loguniform(1e-4, 1e-1)}
        self.optimizer = optimizer

    def _initialize_model(self,X,y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_ratio, random_state=42)
        self.rfplus_model.fit(X_train,y_train)
        if isinstance(self.rfplus_model.estimators_[0], AloGLM):
            self.experts = self.nn.ModuleList([AloTreePlusExpert(self.rfplus_model.estimators_[i],self.rfplus_model.transformers_[i]) for i in range(self.num_experts)])
        else:
            self.experts = self.nn.ModuleList([TreePlusExpert(self.rfplus_model.estimators_[i],self.rfplus_model.transformers_[i]) for i in range(self.num_experts)])
        self.oob_indices_per_expert = [torch.tensor(self.rfplus_model._oob_indices[i])for i in range(self.num_experts)]
        self.gate = GatingNetwork(X.shape[1], self.num_experts,self.noise_epsilon,self.noisy_gating)
       
        train_dataloader = self._get_dataloader(X, y)
        val_dataloader = self._get_dataloader(X_val, y_val)
        return train_dataloader, val_dataloader

    def fit(self,X,y):
        train_dataloader, val_dataloader = self._initialize_model(X, y)
        

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

        # Compute expert outputs
        expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
        expert_outputs = torch.stack([expert(x,index) for i,expert in enumerate(self.experts)], dim=1)
        expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)
        
        return expert_outputs, loss, gating_scores
    

#Define Training Loop
    def train(self, train_loader):
        self.train()
        total_loss = 0
        for batch_idx, (data, target,index) in enumerate(train_loader):
            data, target= data.to(device), target.to(device)
            optimizer.zero_grad()
            output,aux_loss,_ = model(data,index)
            loss = criterion(output, target)
            loss += aux_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))


    
    def cv_squared(self, x):
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + self.gate_epsilon)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)
    
    def _get_dataloader(self, X, y, index):
        dataset = TabularDataset(torch.tensor(X),torch.tensor(y))
        dataloader = DataLoader(dataset, X.shape[0], shuffle=True)
        return dataloader

    @staticmethod
    def tune_model(train_dataset, val_dataset, config):
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=self.max_epochs,
            grace_period=1,
            reduction_factor=2
        )

       






        #     self.experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i]) for i in range(len(rfplus_model.estimators_))])
        # else:
        #     self.experts = nn.ModuleList([TreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i]) for i in range(len(rfplus_model.estimators_))])
    
    # def forward(self, x, index = None):
    #     if self.training:
    #         return self.rfplus_model(x, index)
    #     else:
    #         return self.rfplus_model.predict(x)
        
    # batch_torch_indices = torch.tensor(index) #training indices of elements in batch
#             all_oob_expert_indicator = [torch.isin(batch_torch_indices, expert_oob_indices) for expert_oob_indices in self.oob_indices_per_expert] #indicates which batch elements are oob for each expert
#             all_oob_batch_elements = [batch_torch_indices[oob_expert_indicator] for oob_expert_indicator in all_oob_expert_indicator] #oob elements for each expert
#             all_oob_batch_indices = [torch.nonzero(oob_expert_indicator)for oob_expert_indicator in all_oob_expert_indicator] #indices of oob elements for each expert
#             oob_mask = torch.zeros(x.shape[0],self.num_experts)  
#             for i in range(self.num_experts):
#                 oob_mask[all_oob_batch_indices[i],i] = 1
#             gating_scores = gating_scores * oob_mask 
#         else:
#             _ , topk_indices = gating_scores.topk(min(self.k, self.num_experts), dim=1, sorted=False) #shape (batch_size, k)
#             mask = torch.zeros_like(gating_scores)
#             mask.scatter_(1, topk_indices, 1)   
#             gating_scores = gating_scores * mask
        
        
#         gating_scores = F.softmax(gating_scores, dim=1)
            
#         importance = gating_scores.sum(0)
#         load = self._gates_to_load(gating_scores)
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= self.loss_coef

#         # Compute expert outputs
#         expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
#         expert_outputs = torch.stack([expert(x,index) for i,expert in enumerate(self.experts)], dim=1)
#         expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)
        
#         return expert_outputs, loss, gating_scores





# class RFPlusMoELayer(nn.Module):

#     def __init__(self, input_dim, rfplus_model,k = 1,noise_epsilon = 1e-2, loss_coef = 0.01, noisy_gating = True):
#         """
#         Mixture of Experts Layer for a RF+ model   
        
#         """
#         super(RFPlusMoELayer, self).__init__()
#         self.rfplus_model = rfplus_model
#         num_experts = len(rfplus_model.estimators_)
#         self.num_experts = num_experts  
#         self.experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i]) for i in range(num_experts)])
#         self.oob_indices_per_expert = [torch.tensor(rfplus_model._oob_indices[i])for i in range(num_experts)]
#         self.k = k
#         self.noise_epsilon = noise_epsilon 
#         self.gate = GatingNetwork(input_dim, num_experts,noise_epsilon,noisy_gating)
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))
#         self.softplus = nn.Softplus()
#         self.loss_coef = loss_coef  

    
#     def forward(self, x, index = None): #x has shape (batch_size, input_dim)
       
       
#         # Compute gating scores and get top_k scores
#         gating_scores = self.gate(x)
        

#         if self.training:
#             batch_torch_indices = torch.tensor(index) #training indices of elements in batch
#             all_oob_expert_indicator = [torch.isin(batch_torch_indices, expert_oob_indices) for expert_oob_indices in self.oob_indices_per_expert] #indicates which batch elements are oob for each expert
#             all_oob_batch_elements = [batch_torch_indices[oob_expert_indicator] for oob_expert_indicator in all_oob_expert_indicator] #oob elements for each expert
#             all_oob_batch_indices = [torch.nonzero(oob_expert_indicator)for oob_expert_indicator in all_oob_expert_indicator] #indices of oob elements for each expert
#             oob_mask = torch.zeros(x.shape[0],self.num_experts)  
#             for i in range(self.num_experts):
#                 oob_mask[all_oob_batch_indices[i],i] = 1
#             gating_scores = gating_scores * oob_mask 
#         else:
#             _ , topk_indices = gating_scores.topk(min(self.k, self.num_experts), dim=1, sorted=False) #shape (batch_size, k)
#             mask = torch.zeros_like(gating_scores)
#             mask.scatter_(1, topk_indices, 1)   
#             gating_scores = gating_scores * mask
        
        
#         gating_scores = F.softmax(gating_scores, dim=1)
            
#         importance = gating_scores.sum(0)
#         load = self._gates_to_load(gating_scores)
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= self.loss_coef

#         # Compute expert outputs
#         expert_outputs = torch.zeros_like(gating_scores) #batch size x num_experts
#         expert_outputs = torch.stack([expert(x,index) for i,expert in enumerate(self.experts)], dim=1)
#         expert_outputs = torch.sum(gating_scores * expert_outputs, dim=1)
        
#         return expert_outputs, loss, gating_scores
        

#     def cv_squared(self, x):
       
#         eps = 1e-10
#         if x.shape[0] == 1:
#             return torch.tensor([0], device=x.device, dtype=x.dtype)
#         return x.float().var() / (x.float().mean()**2 + eps)

#     def _gates_to_load(self, gates):
#         return (gates > 0).sum(0)


# def train(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, target,index) in enumerate(train_loader):
#         data, target= data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output,aux_loss,_ = model(data,index)
#         loss = criterion(output, target)
#         loss += aux_loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))


# def val(model, device, val_loader, criterion):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data, target, _ in val_loader:
#             data, target = data.to(device), target.to(device)
#             output,_ ,_= model(data)
#             test_loss += criterion(output, target).item()
#     test_loss /= len(val_loader)
#     print('Validation set: Average loss: {:.4f}\n'.format(test_loss))

    
if __name__ == "__main__":

    RF_plus_MOE = _RandomForestPlusRMOE()


    # #Load Data 
    # suite_id = 353
    # benchmark_suite = openml.study.get_suite(suite_id)
    # task_ids = benchmark_suite.tasks

    # #task_id =  task_ids[3] 
    # task_id =  361256
    # random_state = 10
    # print(f"Task ID: {task_id}")
    # task = openml.tasks.get_task(task_id)
    # dataset_id = task.dataset_id
    # dataset = openml.datasets.get_dataset(dataset_id)


    


    # # Split data into train, validation, and test sets
    # num_train = 500
    # X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
    # #X,y,f = imodels.get_clean_dataset("fico")
    # X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.4,random_state=random_state)
    # X_train, y_train = X_train_full[:num_train], y_train_full[:num_train]
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state+1)
    # #X_val, y_val = X_val[:200], y_val[:200]

    # print(f"X_validation has shape: {X_val.shape}")
    
    # X_train_and_val = np.concatenate((copy.deepcopy(X_train),copy.deepcopy(X_val)),axis = 0)
    # y_train_and_val = np.concatenate((copy.deepcopy(y_train),copy.deepcopy(y_val)),axis = 0)

    # #Train a RF plus model
    # n_estimators = 256
    # min_samples_leaf = 5

    # rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33,random_state=0)

    # rf.fit(X_train,y_train)

    # rf_train_val = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33,random_state=0)
    # rf_train_val.fit(X_train_and_val,y_train_and_val)

    # rfplus_model = RandomForestPlusRegressor(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
    #                                                                max_features=0.33,random_state=0),
    #                                                                prediction_model= AloElasticNetRegressorCV(n_splits=0,l1_ratio=[0.0]),fit_on = "all")  
    # rfplus_model.fit(X_train,y_train,n_jobs=-1)

    # # Define the MoEModel class
    # input_dim = X.shape[1]
    # output_dim = 1
    # k = 256
  
    # rfMOE = RFPlusMoELayer(input_dim, rfplus_model, k,noise_epsilon = 1e-2, noisy_gating = False)
    

    # #Define device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # #Define Train DataLoader and Val DataLoader
    # batch_size = X_train.shape[0]
    
    
    # trainset = TabularDataset(torch.tensor(X_train), torch.tensor(y_train))
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)


    # valset = TabularDataset(torch.tensor(X_val), torch.tensor(y_val))
    # valloader = torch.utils.data.DataLoader(trainset, batch_size=X_val.shape[0],shuffle=False)

    # # Training hyperparameters
    # num_epochs = 1
    # lr = 1e-2
    # optimizer = torch.optim.Adam(rfMOE.parameters(), lr=lr)
    # criterion = nn.MSELoss()

    # print("Before MOE")
    # val(rfMOE, device, valloader, criterion)
    
    # for epoch in range(1, num_epochs+1):
    #     print("Starting Epoch: ",epoch)  
    #     train(rfMOE, device, trainloader, optimizer, criterion, epoch)
    #     val(rfMOE, device, valloader, criterion)

    
    # #Test the model
    # rfMOE.eval()
    # rfplus_preds = rfplus_model.predict(X_test)
    # rf_preds = rf.predict(X_test)
    # rf_train_val_preds = rf_train_val.predict(X_test)
    # rfMOE_preds,_,gating_scores = rfMOE(torch.tensor(X_test,dtype=torch.float32))
    # rfMOE_preds = rfMOE_preds.cpu().detach().numpy()
    # metrics = [root_mean_squared_error, r2_score, mean_absolute_error]

      

    # for metric in metrics:
    #     print(metric.__name__)
    #     print("RF (Train) Model: ",metric(y_test,rf_preds))
    #     print("RF (Train+Val) Model: ",metric(y_test,rf_train_val_preds))
    #     print("RF+ Model: ",metric(y_test,rfplus_preds))
    #     print("RF+ MoE Model: ",metric(y_test,rfMOE_preds))
    #     print("\n")

    # # Average gating scores




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
