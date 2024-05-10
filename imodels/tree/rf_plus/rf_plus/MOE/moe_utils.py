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

    def __init__(self,estimator_,transformer_,):
        super(TreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        self.transformer_ = transformer_
        
    
    def forward(self,x,index = None): #x has shape (batch_size, input_dim)
        
        x = x.numpy()
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        x = self.transformer_.transform(x).get_all_data()
        out = self.estimator_.predict(x)
        return torch.tensor(out).float()



class AloTreePlusExpert(nn.Module):
    def __init__(self,estimator_,transformer_,):
        super(AloTreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        self.transformer_ = transformer_
    
    def forward(self,x,index = None): #x has shape (batch_size, input_dim)
        
        x = x.numpy()
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        if self.training:
            x = self.transformer_.transform(x).get_all_data()
            x1 = np.hstack((x,np.ones((x.shape[0],1))))
            batch_loo_coef = self.estimator_.loo_coefficients_[index.numpy(),:]
            out = self.estimator_.inv_link_fn(np.sum(x1*batch_loo_coef,axis = 1))
        else:
            x = self.transformer_.transform(x).get_all_data()
            out = self.estimator_.inv_link_fn(x@self.estimator_.coefficients_ + self.estimator_.intercept_) 
        
        return torch.tensor(out).float()
    


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
    