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
set_seed()

# loading the data and getting familiar
file_path = "/home/ahmad_rahmani/Github_Repositories/multivariate_lstm/BTC-USD.csv" 
df = pd.read_csv(file_path) 
df['Date'] = pd.to_datetime(df['Date']) 
df.drop(columns='Adj Close', inplace=True) 
df.set_index('Date', inplace=True) 
print(df.head(5)) 
data = df.to_numpy() 
print("Data shape: ",data.shape)

# plotting the labels
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
label_column = 3 #4th column is the labels
features, labels = data_setup.sequence_generator(timeseries=data, lookback_period=lookback_period, 
                                                 pred_length=prediction_length, label_column=label_column) 
print('Features shape: ',features.shape)  # from here we are in tensor domain 
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
input_size = 5 # number of features
hidden_size = 80  # number of hidden units in an LSTM layer
num_layers = 1  # number of LSTM layers
output_size = 1 # number of predicted variables, here "Close" only
prediction_length = 10 
torch.manual_seed(seed=42)
LSTM_nn = model.LSTM_model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, pred_len=prediction_length) 

# model output shape !! 
input = torch.randn(32,50,5)
LSTM_nn.eval() 
with torch.inference_mode(): 
    output = LSTM_nn(input)
print('Output shape: ', output.shape) 

# creating loss functions and optimizer 
loss_fn = torch.nn.MSELoss() 
learning_rate = 0.01 
optimizer_LSTM = optim.Adam(LSTM_nn.parameters(), learning_rate)  

# training LSTM 
model_name_LSTM = "LSTM_model"
training_epochs_LSTM = 20 
training_results_LSTM = engine.train_model(model=LSTM_nn, train_dataloader=train_dataloder, valid_dataloader=validation_dataloader,
                                      optimizer=optimizer_LSTM, loss_fn=loss_fn, epochs=training_epochs_LSTM, model_name=model_name_LSTM) 

# testing LSTM 
test_results_LSTM, test_predictions, tests  = engine.test_step(model=LSTM_nn, dataloader=testing_dataloader, loss_fn=loss_fn) 
print("prediction tensor shape:", test_predictions.shape) 
print("test tensor shape: ", tests.shape)

# plot of the testing results  
# Convert torch tensors to numpy arrays first
preds = test_predictions.numpy()  # shape: (404, 10)
trues = tests.numpy()            # shape: (404, 10) 
# Flatten the 2D array into shape (404 * 10, ) # to have all the predictions as a single vector
preds_flat = preds.reshape(-1, 1)
trues_flat = trues.reshape(-1, 1)
# Prepare dummy 5D data for inverse transform
# We'll insert our close prices in column index 3 (the 'Close' column)
dummy_preds = np.zeros((preds_flat.shape[0], 5))
dummy_trues = np.zeros((trues_flat.shape[0], 5))
dummy_preds[:, label_column] = preds_flat[:, 0]  # column index 3 = 'Close'
dummy_trues[:, label_column] = trues_flat[:, 0]
# Perform inverse transform
inv_preds = scaler.inverse_transform(dummy_preds)[:, 3]  # extract only the 'Close' column
inv_trues = scaler.inverse_transform(dummy_trues)[:, 3]
# Reshape to original prediction shape (N, 10)
inv_preds = inv_preds.reshape(preds.shape)
inv_trues = inv_trues.reshape(trues.shape)

# averaging alignment strategy
total_len = inv_preds.shape[0] + inv_preds.shape[1]
pred_series = np.zeros((total_len,))
true_series = np.zeros((total_len,))
count = np.zeros((total_len,))
for i in range(inv_preds.shape[0]):
    start_idx = i
    end_idx = i + prediction_length
    pred_series[start_idx:end_idx] += inv_preds[i]
    true_series[start_idx:end_idx] += inv_trues[i]
    count[start_idx:end_idx] += 1
count[count == 0] = 1
pred_series /= count
true_series /= count

# Plot
plt.figure(figsize=(14, 5))
plt.plot(true_series, label="True (USD)")
plt.plot(pred_series, label="Predicted (USD)")
plt.legend()
plt.title("Predicted vs True Close Prices (Inverse Transformed)")
plt.xlabel("Time steps")
plt.ylabel("Price [USD]")
plt.grid(True)
plt.show()