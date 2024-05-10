

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
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
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


class AloTreePlusExpert(nn.Module):
    def __init__(self,estimator_,transformer_,):
        super(AloTreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        self.transformer_ = transformer_
        #self.fc = nn.Linear(self.transformer_.transformed_dim,self.transformer_.transformed_dim)
    
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

        #out = self.estimator_.predict_loo(x,index.numpy())

class TreePlusExpert(nn.Module):

    def __init__(self,estimator_,transformer_,):
        super(TreePlusExpert, self).__init__()
        
        self.estimator_ = estimator_
        self.transformer_ = transformer_
        self.fc = nn.Linear(self.transformer_.transformed_dim,self.transformer_.transformed_dim) 

        nn.init.eye_(self.fc.weight)  
        nn.init.zeros_(self.fc.bias)
        
    
    def forward(self,x,index = None): #x has shape (batch_size, input_dim)
        
        x = x.numpy()
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        x = self.transformer_.transform(x).get_all_data()
        # x = self.fc(torch.tensor(x).float())
        # y = x.clone().detach().numpy() 
        # out = self.estimator_.predict(y)
        out = self.estimator_.predict(x)
        return torch.tensor(out).float()




class GatingNetwork(nn.Module):
    def __init__(self, input_dim,num_experts,noise_epsilon=1e-2,noisy_gating = True):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts 
        self.first_gate = nn.Linear(input_dim, input_dim)  
        #self.batchnorm = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.second_gate = nn.Linear(input_dim, input_dim)
        self.relu2 = nn.ReLU()
        self.output_gate = nn.Linear(input_dim, num_experts)
        self.noisy_gating = noisy_gating

       
        nn.init.constant_(self.output_gate.weight,1.0/num_experts)
        nn.init.zeros_(self.output_gate.bias)
        
    def forward(self, x):
        gating_scores = self.first_gate(x)
        #gating_scores = self.batchnorm(gating_scores)
        gating_scores = self.relu(gating_scores)
        #gating_scores = self.second_gate(gating_scores)
        # gating_scores = self.relu2(gating_scores)
        gating_scores = self.output_gate(gating_scores)
        if self.noisy_gating:
            pass
        return gating_scores
    
    
class RFPlusMoELayer(nn.Module):

    def __init__(self, input_dim, rfplus_model,k = 0.25,noise_epsilon = 1e-2, loss_coef = 0.01, noisy_gating = True):
        """
        Mixture of Experts Layer for a RF+ model   
        
        """
        super(RFPlusMoELayer, self).__init__()
        self.rfplus_model = rfplus_model
        num_experts = len(rfplus_model.estimators_)
        self.num_experts = num_experts  
        self.experts = nn.ModuleList([AloTreePlusExpert(rfplus_model.estimators_[i],rfplus_model.transformers_[i]) for i in range(num_experts)])
        self.oob_indices_per_expert = [torch.tensor(rfplus_model._oob_indices[i])for i in range(num_experts)]
        self.k = k
        self.noise_epsilon = noise_epsilon 
        self.gate = GatingNetwork(input_dim, num_experts,noise_epsilon,noisy_gating)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softplus = nn.Softplus()
        self.loss_coef = loss_coef  

    
    def forward(self, x, index = None): #x has shape (batch_size, input_dim)
       
       
        # Compute gating scores and get top_k scores
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
        

    def cv_squared(self, x):
       
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

#Define Training Loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
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


def val(model, device, val_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target, _ in val_loader:
            data, target = data.to(device), target.to(device)
            output,_ ,_= model(data)
            test_loss += criterion(output, target).item()
    test_loss /= len(val_loader)
    print('Validation set: Average loss: {:.4f}\n'.format(test_loss))

    
if __name__ == "__main__":

    #Load Data 
    suite_id = 353
    benchmark_suite = openml.study.get_suite(suite_id)
    task_ids = benchmark_suite.tasks

    #task_id =  task_ids[3] 
    task_id =  361244
    random_state = 1
    print(f"Task ID: {task_id}")
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    


    # Split data into train, validation, and test sets
    num_train = 500
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute,dataset_format="array")
    #X,y,f = imodels.get_clean_dataset("fico")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.4,random_state=random_state)
    X_train, y_train = X_train_full[:num_train], y_train_full[:num_train]
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state+1)
    #X_val, y_val = X_val[:200], y_val[:200]

    print(f"X_validation has shape: {X_val.shape}")
    
    X_train_and_val = np.concatenate((copy.deepcopy(X_train),copy.deepcopy(X_val)),axis = 0)
    y_train_and_val = np.concatenate((copy.deepcopy(y_train),copy.deepcopy(y_val)),axis = 0)

    #Train a RF plus model
    n_estimators = 256
    min_samples_leaf = 5

    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33,random_state=0)

    rf.fit(X_train,y_train)

    rf_train_val = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=0.33,random_state=0)
    rf_train_val.fit(X_train_and_val,y_train_and_val)

    rfplus_model = RandomForestPlusRegressor(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                                                   max_features=0.33,random_state=0),
                                                                   prediction_model= AloElasticNetRegressorCV(n_splits=0,l1_ratio=[0.0]),fit_on = "all")  
    rfplus_model.fit(X_train,y_train,n_jobs=-1)

    # Define the MoEModel class
    input_dim = X.shape[1]
    output_dim = 1
    k = 256
  
    rfMOE = RFPlusMoELayer(input_dim, rfplus_model, k,noise_epsilon = 1e-2, noisy_gating = False)
    

    #Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Define Train DataLoader and Val DataLoader
    batch_size = X_train.shape[0]
    
    
    trainset = TabularDataset(torch.tensor(X_train), torch.tensor(y_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)


    valset = TabularDataset(torch.tensor(X_val), torch.tensor(y_val))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=X_val.shape[0],shuffle=False)

    # Training hyperparameters
    num_epochs = 2
    lr = 1e-2
    optimizer = torch.optim.Adam(rfMOE.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Before MOE")
    val(rfMOE, device, valloader, criterion)
    
    for epoch in range(1, num_epochs+1):
        print("Starting Epoch: ",epoch)  
        train(rfMOE, device, trainloader, optimizer, criterion, epoch)
        val(rfMOE, device, valloader, criterion)

    
    #Test the model
    rfMOE.eval()
    rfplus_preds = rfplus_model.predict(X_test)
    rf_preds = rf.predict(X_test)
    rf_train_val_preds = rf_train_val.predict(X_test)
    rfMOE_preds,_,gating_scores = rfMOE(torch.tensor(X_test,dtype=torch.float32))
    rfMOE_preds = rfMOE_preds.cpu().detach().numpy()
    metrics = [root_mean_squared_error, r2_score, mean_absolute_error]

      

    for metric in metrics:
        print(metric.__name__)
        print("RF (Train) Model: ",metric(y_test,rf_preds))
        print("RF (Train+Val) Model: ",metric(y_test,rf_train_val_preds))
        print("RF+ Model: ",metric(y_test,rfplus_preds))
        print("RF+ MoE Model: ",metric(y_test,rfMOE_preds))
        print("\n")

    # Average gating scores




    # rows_to_plot = [10,11,12]
    # # Create a figure and axes
    # fig, ax = plt.subplots()
    # gating_scores = gating_scores.detach().numpy()
    # # Set the width of each bar
    # bar_width = 0.8 / len(rows_to_plot)

    # # Iterate over the selected rows and plot them as bar plots
    # for i, row_idx in enumerate(rows_to_plot):
    #     row_data = gating_scores[row_idx]
    #     x = np.arange(gating_scores.shape[1]) + i * bar_width
    #     ax.bar(x, row_data, width=bar_width, label=f'Row {row_idx}')

    # # Set the x-tick labels and positions
    # x_ticks = np.arange(gating_scores.shape[1]) + bar_width * (len(rows_to_plot) - 1) / 2
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([f'{i}' for i in range(gating_scores.shape[1])])

    # # Add a legend
    # ax.legend()

    # #plt.show()


 # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Gradients for {name}:")
        #         print(param.grad)
 # if (self.training) and (self.noisy_gating):
        #     raw_noise_stddev = self.noise_param(x)
        #     noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
        #     gating_scores = clean_gating_scores + torch.randn_like(clean_gating_scores) * noise_stddev
        # else:
        #     gating_scores = clean_gating_scores



# class SmoothStep(nn.Module):
#     def __init__(self, gamma=1.0):
#         super(SmoothStep, self).__init__()
#         self._lower_bound = -gamma / 2
#         self._upper_bound = gamma / 2
#         self._a3 = -2 / (gamma**3)
#         self._a1 = 3 / (2 * gamma)
#         self._a0 = 0.5

#     def forward(self, inputs):
#         return torch.where(inputs <= self._lower_bound,torch.zeros_like(inputs),
#             torch.where(inputs >= self._upper_bound,torch.ones_like(inputs),
#                 self._a3 * (inputs**3) + self._a1 * inputs + self._a0,),)


# class EntropyRegularizer(nn.Module):
#     def __init__(self, schedule_fn=lambda x: 1e-6):
#         super(EntropyRegularizer, self).__init__()
#         self._num_calls = nn.Parameter(torch.zeros(1), requires_grad=False)
#         self._schedule_fn = schedule_fn

#     def forward(self, inputs):
#         self._num_calls += 1
#         reg_param = self._schedule_fn(self._num_calls)
#         entropy = -torch.sum(inputs * torch.log(inputs + EPSILON))
#         return reg_param * entropy
