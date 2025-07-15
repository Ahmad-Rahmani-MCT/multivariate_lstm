#!/usr/bin/env python3

# sets up the Neural Network class
import torch 
import torch.nn as nn 

class LSTM_model(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_len): 
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.output_size = output_size
        self.pred_len = pred_len 
        # LSTM layer 
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)  # the LSTM layer gets inputs of the shape (batch_size, sequence_length, input_size), so we dont flatten it
        # fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size) 
        # relu layer
        self.relu = nn.ReLU() 

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.lstm(x, (h0, c0))
        #out = self.relu(out) 
        out = out[:, -self.pred_len:, :] 
        out = self.fc(out) 
        return out 