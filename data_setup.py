#!/usr/bin/env python3

# This scripts contains functions to set up the data by loading, preprocessing, creating a sequence and dataset and dataloader
import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split
 

def sequence_generator(timeseries: np.array, lookback_period: int, pred_length: int, label_column: int): 
    """
    Creates a sequence of features and labels corresponding the to the lookback_period, prediction length and the label column from a time
    series data. RETURNS A TORCH TENSOR
    """
    features =[] 
    labels = [] 
    for i in range (len(timeseries)): 
        feature_index_end = i + lookback_period 
        label_index_start = feature_index_end
        label_index_end = label_index_start + pred_length 
        if label_index_end >= len(timeseries) : break 
        feature_sequence = timeseries[i:feature_index_end,:]
        label_sequence = timeseries[label_index_start:label_index_end, label_column]  
        features.append(feature_sequence) 
        labels.append(label_sequence) 
    features = np.array(features)  #convert the list to a numpy aray instead of having a list of arrays in which conversion to a tensor will be slow
    labels = np.array(labels) 
    return torch.Tensor(features), torch.Tensor(labels) 
 

def train_valid_test_split(features: torch.Tensor, labels: torch.Tensor, train_percent: int,
                                 valid_percent: int, test_percent: int): 
    """
    turns a feature and label tensor into training, validation and testing splits. 
    """ 
    data_size = len(features) 
    train_size = int(train_percent * data_size) 
    valid_size = int(valid_percent * data_size)
    test_size = data_size - train_size - valid_size 
    train_dataset = TensorDataset(features[0 : train_size, :, :],labels[0:train_size, :]) 
    validation_dataset = TensorDataset(features[train_size:train_size + valid_size, :, :], labels[train_size:train_size + valid_size, :]) 
    testing_dataset = TensorDataset(features[train_size + valid_size:, :, :], labels[train_size + valid_size:, :]) 
    return train_dataset, validation_dataset, testing_dataset  

 
def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size: int): 
    """
    Creates dataloaders from a subset object according to specified training, validation and testing sizes and batch size 
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    return train_loader, valid_loader, test_loader 