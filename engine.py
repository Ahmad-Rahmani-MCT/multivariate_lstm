#!/usr/bin/env python3
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# trains the neural network 

def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
               optimizer: optim.Optimizer):
    """
    Trains a pytorch model for a single epoch 
    """
    #put model in train mode
    model.train()  
    #setting up train loss and accuracy values 
    train_loss = 0  
    #looping through the dataloader batches 
    for batch, (X,y) in enumerate(dataloader): #enumarate returns (index, value) -> index = batch & value = (X,y) 
        #forwrard pass 
        y_pred = model(X) 
        y_pred = y_pred.squeeze(-1) #to squeeze the output dimension from [batch_size, 1, 6] tp [batch_size,6] to match the y from dataloader
        #calculate loss and accmulate the loss 
        loss = loss_fn(y_pred, y) 
        train_loss += loss.item()  #.item() to get python scalar from the tensor 
        #optimizer zero grad 
        optimizer.zero_grad() 
        #loss backward 
        loss.backward() # computes the gradients of the loss with respect to all the model parameters and stores in .grad attribute of the model 
        #optimizer step 
        optimizer.step() 
        
    #adjusting the loss 
    train_loss = train_loss / len(dataloader)  
    return train_loss


def valid_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module): 
    """
    validates a pytorch model for a single epoch 
    """
    #put model in eval mode 
    model.eval() 
    #setup test loss values
    valid_loss = 0.0
    #turn on inference context manager 
    with torch.inference_mode(): 
        #loop through the dataloader 
        for batch, (X,y) in enumerate(dataloader): 
            #forward pass 
            y_pred_valid = model(X) 
            y_pred_valid = y_pred_valid.squeeze(-1)
            # calculate and accumulate loss 
            loss = loss_fn(y_pred_valid, y) 
            valid_loss += loss.item() 
    #adjusting the metrics 
    valid_loss = valid_loss / len(dataloader) #average
    return valid_loss

def train_model(model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epochs: int, model_name: str): 
    """
    Passes a pytorch model through train_step and valid_step for a number of epochs 
    """ 

    #create empty result dictionary 
    results = {"train_loss": [],
               "valid_loss": []}
    # training loop 
    for epoch in range(epochs): 
        train_loss = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer) 
        valid_loss = valid_step(model=model, dataloader=valid_dataloader, loss_fn=loss_fn) 
        #print out whats happpening 
        print(
            f"{model_name} ! "
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | " # :.4f keep 4 decimal places 
            f"Validation_Loss: {valid_loss:.4f} | " 
        )
        # update the results dictionary 
        results["train_loss"].append(train_loss) 
        results["valid_loss"].append(valid_loss) 

    return results

def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module): 
    """
    tests a pytorch model for a single epoch 
    """
    #put model in eval mode 
    model.eval() 
    #setup test loss values
    test_loss = 0.0 
    # prediction and testing lists
    predictions = [] 
    tests = []
    #turn on inference context manager 
    with torch.inference_mode(): 
        #loop through the dataloader 
        for batch, (X,y) in enumerate(dataloader): 
            #forward pass 
            y_test_valid = model(X) 
            y_test_valid = y_test_valid.squeeze(-1)
            # calculate and accumulate loss 
            loss = loss_fn(y_test_valid, y) 
            test_loss += loss.item() 
            #updating the list
            predictions.append(y_test_valid) 
            tests.append(y) 
    #adjusting the metrics 
    test_loss = test_loss / len(dataloader) #average
    print('Test Loss: ', test_loss)
    return test_loss, torch.cat(predictions, dim=0), torch.cat(tests, dim=0) 