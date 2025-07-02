 
 

import numpy as np
import torch
from marginalTailAdaptiveFlow.utils.flows import experiment

 


def train_test_split(data, test_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    B = data.shape[0]
    indices = np.random.permutation(B)
    test_size = int(B * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return data[train_indices], data[test_indices]

 


def Gen_Dataset(Dataset,seed=42):

    if Dataset=="studentT": #perfect
        dof_og=2
        full_data=torch.tensor(np.load('data\\ST\\ST2.npy'))
        name=str(dof_og)+'_studentT'
        full_data_train,full_data_test=train_test_split(full_data,seed=seed,test_ratio=0.5)
        full_data_val=full_data_train
        dimension=2
        num_heavy=2
    elif Dataset=="pareto": #perfect
        name='pareto'

        dimension=20
        num_heavy=20
        full_data = np.load('data\\pareto\\pareto_20d_data.npy').reshape(-1,20) 
        full_data_train,full_data_test=train_test_split(full_data,seed=seed)
        full_data_val=full_data_train
   
    elif Dataset=='funnel':
        name='funnel'
        dimension=2
        num_heavy=1
        n_hidden_layers=2
        full_data=torch.tensor(np.load('data\\Neal_funnel\\funnel_data.npy'))
        full_data_train,full_data_test=train_test_split(full_data,seed=seed)
        full_data_val=full_data_train
        print(full_data_train.shape)

            
    elif Dataset=="Stock":
        full_data_train = np.load('data\\stock\\stock_data_train_20.npy').reshape(-1,20)# 1200,20
        full_data_test = np.load('data\\stock\\stock_data_test_20.npy').reshape(-1,20)# 282,20
        full_data_val=full_data_train
        dimension=20
        num_heavy=20
        name='stock'


    elif Dataset=='copula':
        name='copula'
        exp = experiment("copula")
        exp.load_data(8, num_heavy=4, df=2, seed=seed)
        num_heavy=4
        full_data_train = exp.data_train
        full_data_test = exp.data_test
        full_data_val = exp.data_val
        dimension=8
    elif Dataset=='weather':
        name='weather'
        num_heavy=174
        exp2 = experiment("copula")
        exp2.load_data('weather', estimate_tails=True)
        exp2.get_weather_tails()
        full_data_train = exp2.data_train
        full_data_test = exp2.data_test
        full_data_val = exp2.data_val
        dimension=412
    params={'name':name,'dimension':dimension,'num_heavy':num_heavy}
    data={'train':full_data_train,'val':full_data_val,'test':full_data_test}
    return params,data







