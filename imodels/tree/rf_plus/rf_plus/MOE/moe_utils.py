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
from sklearn.model_selection import train_test_split

#RF plus imports
import imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import _RandomForestPlus, RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloElasticNetRegressorCV, AloLOL2Regressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier, AloLogisticElasticNetClassifierCV, AloSVCRidgeClassifier

#Teesting imports
import openml
import time

import torch
torch.manual_seed(0)




class TabularDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float(), index

    def __len__(self):
        return len(self.x)
    

class TreePlusExpert(nn.Module):

    def __init__(self,estimator_,input_dim):
        super(TreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        treeplus_coefficients = torch.from_numpy(estimator_.coefficients_).float()  # Ensure the tensor type is appropriate (float for most cases)
        treeplus_intercept = torch.from_numpy(np.array([estimator_.intercept_])).float()
        treeplus_layer = nn.Linear(input_dim,1)
        with torch.no_grad():
            treeplus_layer.weight.copy_(treeplus_coefficients)
            treeplus_layer.bias.copy_(treeplus_intercept)
            treeplus_layer.weight.requires_grad = False
            treeplus_layer.bias.requires_grad = False   
        self.treeplus_layer = treeplus_layer

        
    
    def forward(self,x,index = None): #x has shape (batch_size, input_dim)
        
        out = self.treeplus_layer(x).squeeze(1)
        return out



class AloTreePlusExpert(nn.Module):
    def __init__(self,estimator_,input_dim):
        super(AloTreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        treeplus_coefficients = torch.from_numpy(estimator_.coefficients_).float()  # Ensure the tensor type is appropriate (float for most cases)
        treeplus_intercept = torch.from_numpy(np.array([estimator_.intercept_])).float()
        treeplus_layer = nn.Linear(input_dim,1)
        with torch.no_grad():
            treeplus_layer.weight.copy_(treeplus_coefficients)
            treeplus_layer.bias.copy_(treeplus_intercept)
            treeplus_layer.weight.requires_grad = False
            treeplus_layer.bias.requires_grad = False   
        self.treeplus_layer = treeplus_layer

        treeplus_loo_coefficients = torch.from_numpy(estimator_.loo_coefficients_[:,:-1]).float()  
        treeplus_loo_intercept = torch.from_numpy(estimator_.loo_coefficients_[:,-1]).float()    
        self.treeplus_loo_layer = torch.nn.Parameter(treeplus_loo_coefficients,requires_grad=False)
        self.treeplus_loo_intercept = torch.nn.Parameter(treeplus_loo_intercept,requires_grad=False)

        
    def forward(self,x,index = None): #x has shape (batch_size, input_dim)
        
        if self.training:
            out = torch.sum(x*self.treeplus_loo_layer[index,:],dim = 1) + self.treeplus_loo_intercept[index]
        else:        
            out = self.treeplus_layer(x).squeeze(1)
        return out
        
    


class GatingNetwork(nn.Module):
    def __init__(self, input_dim,num_experts,noise_epsilon=1e-2,noisy_gating = True):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts 
        self.first_gate = nn.Linear(input_dim, input_dim)  
        self.relu = nn.ReLU()
        self.second_gate = nn.Linear(input_dim, input_dim)
        self.relu2 = nn.ReLU()
        self.output_gate = nn.Linear(input_dim, num_experts)
        self.noisy_gating = noisy_gating

       
        nn.init.constant_(self.output_gate.weight,1.0/num_experts)
        nn.init.zeros_(self.output_gate.bias)
        
    def forward(self, x):
        gating_scores = self.first_gate(x)
        gating_scores = self.relu(gating_scores)
        gating_scores = self.second_gate(gating_scores)
        gating_scores = self.relu2(gating_scores)
        gating_scores = self.output_gate(gating_scores)
        if self.noisy_gating:
            pass
        return gating_scores
    





        #fc = nn.Linear(self.transformer_.transformed_dim,self.transformer_.transformed_dim)
        # torch.nn.init.eye_(fc.weight)
        # torch.nn.init.zeros_(fc.bias)
        # self.fc = fc
