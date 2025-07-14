import torch 
import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler 
import data_setup, model, engine

# setting seed for reproducability 
def set_seed(seed=42): 
    torch.manual_seed(seed=seed) 
    np.random.seed(seed=seed)
    random.seed(seed) 

set_seed 

# loading the data and setting up 
file_path = "/home/ahmad_rahmani/Github_Repositories/multivariate_lstm/BTC-USD.csv" 
df = pd.read_csv(file_path) 
df['Date'] = pd.to_datetime(df['Date']) 
df.drop(columns='Adj Close', inplace=True) 
df.set_index('Date', inplace=True) 
print(df.head(5)) 
data = df.to_numpy() 
print("Data shape: ",data.shape)

# plots 
plt.plot(data[:,3]) 
plt.xlabel("Time") 
plt.ylabel("price [USD]")  
plt.title('Bitcoin Prices vs Years')
# plt.show() 

# scaling the data 
scaler = MinMaxScaler() 
data = scaler.fit_transform(data) 

# generating the time series sequence, and obtaining the features and labels tensors
lookback_period = 50  
prediction_length = 10
features, labels = data_setup.sequence_generator(timeseries=data, lookback_period=lookback_period, 
                                                 pred_length=prediction_length) 
print('Features shape: ',features.shape) 
print('Labels shape: ',labels.shape) 

# train, validation and testing split 
train_percent = 0.75 
validation_percent = 0.10 
testing_percent = 0.15
train_dataset, validation_dataset, testing_dataset = data_setup.train_valid_test_split(features=features, labels=labels, train_percent=train_percent,
                                                                                       valid_percent=validation_percent, test_percent=testing_percent) 
print('train_dataset length: ',len(train_dataset)) 
print('validation_dataset length: ',len(validation_dataset)) 
print('testing_dataset length: ',len(testing_dataset)) 

# turning the datasets into dataloader 
batch_size = 32 
train_dataloder, validation_dataloader, testing_dataloader = data_setup.create_dataloaders(train_dataset=train_dataset, valid_dataset=validation_dataset,
                                                                                           test_dataset=testing_dataset, batch_size=batch_size) 
print('train dataloader length: ',len(train_dataloder)) 
print('validation dataloader length: ',len(validation_dataloader)) 
print('testing dataloader length: ',len(testing_dataloader)) 
for X_batch, y_batch in train_dataloder: 
    print('shape of a single feature element in a batch', X_batch.shape) 
    print('shape of a single label element in a batch', y_batch.shape)
    break 

# instantiating an instance of the model class 
input_size = 5 
hidden_size = 80 
num_layers = 1 
output_size = 1 
prediction_length = 10 
LSTM_nn = model.LSTM_model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, pred_len=prediction_length) 

# creating loss functions and optimizer 
loss_fn = torch.nn.MSELoss() 
learning_rate = 0.01 
optimizer_LSTM = optim.Adam(LSTM_nn.parameters(), learning_rate)  

# training LSTM 
model_name_LSTM = "LSTM_model"
training_epochs_LSTM = 20 
training_results_LSTM = engine.train_model(model=LSTM_nn, train_dataloader=train_dataloder, valid_dataloader=validation_dataloader,
                                      optimizer=optimizer_LSTM, loss_fn=loss_fn, epochs=training_epochs_LSTM, model_name=model_name_LSTM) 