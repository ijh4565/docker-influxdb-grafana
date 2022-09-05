from typing import Callable,Dict,Optional,Tuple
from typing import List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import Metrics
import numpy as np
import sys
import math
import time
device = torch.device("cpu")

import warnings
warnings.filterwarnings( 'ignore' )

# Convert data to form of sequence data
def seq_data(x, sequence_length):
    
    x_seq = []
    y_seq = []
    for i in range(len(x)-sequence_length-1):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(x[i+sequence_length])
    
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1]) 
# LSTM Model
class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128,1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        _, (hn,cn) = self.lstm(x, (h0, c0)) 
        
        out = hn.view(-1, self.hidden_size) 
        out=self.relu(out)
        out = self.fc(out)
        out = self.fc2(out)
        return out
# To evaluate the model performance
def accuracy(dataloader, model):
    n = len(dataloader)
    running_loss_arr = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        running_loss = 0.0
        model.eval() 
        for data in dataloader:
            inputs, labels = data # Labels = Ground truth Y
            outputs = model(inputs) # 예측값 Predictied Y 
            loss = criterion(outputs, labels)
            loss = math.sqrt(loss)/torch.mean(outputs)
            running_loss += loss.item()
        running_loss_arr.append(running_loss/n)
        
    model.train()
    return running_loss_arr

def report(dataloader, model):
    n = len(dataloader) # batch size ? 
    real_y = []
    predicted_y = []

    with torch.no_grad():
        model.eval() 
        for data in dataloader:
            inputs, labels = data # Labels = Ground truth Y
            outputs = model(inputs) # 예측값 Predictied Y 
            real_y.append(labels)
            predicted_y.append(outputs)
        
    model.train()
    return real_y, predicted_y

# Read data
with open('/home/proj01/gangmin/test/datas.pkl','rb') as f:
    data = pickle.load(f)
columns = data.columns
total_loss=[]
total_time=[]
for col_index, col in enumerate(columns):
  
  if col_index<2 or col_index == 6 or col_index == 9:
    continue
  if col_index ==14:
    break
  
  X = data[columns[col_index]].values
  data_name = columns[col_index]

  split = int(len(X)*0.7)
  sequence_length = 8
  X = np.expand_dims(X, axis=1)
  x_seq, y_seq = seq_data(X, sequence_length)

  # Split entire dataset into train and test set
  x_train_seq = x_seq[:split]
  y_train_seq = y_seq[:split]
  x_test_seq = x_seq[split:]
  y_test_seq = y_seq[split:]
    
  # Convert to Tensor
  train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
  test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)
  total = torch.utils.data.TensorDataset(x_seq, y_seq)

  # Generate Data Loader
  batch_size = 128

  train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
  data_loader = torch.utils.data.DataLoader(dataset=total, batch_size=batch_size, shuffle=True) # total data

  # Write final evaluation results
  with open('weight_'+data_name+'.pkl','rb') as f:
            eval_model = pickle.load(f)
  avg_loss=[]
  avg_time=[]

  for _ in range(5):
    s = time.time()
    test_loss = accuracy(test_loader,eval_model)
    avg_loss.append(test_loss)
    avg_time.append(time.time()-s)

  
  real_y, predicted_y = report(data_loader,eval_model)
  total_loss = accuracy(data_loader,eval_model)
  with open('./realY/realY_'+data_name+'.pickle', 'wb') as f:
    pickle.dump(real_y, f, pickle.HIGHEST_PROTOCOL)
  with open('./predictedY/predictedY_'+data_name+'.pickle', 'wb') as f:
    pickle.dump(predicted_y, f, pickle.HIGHEST_PROTOCOL)
  print("TOTAL RMSE LOSS VALUE : ",total_loss)

  if col_index ==2:
    f= open('prediction_result.txt','w')
    f.write('############# Average prediction result #############\n')
  else:
    f= open('prediction_result.txt','a')
  f.write('data :'+'%15s'%data_name)
  f.write('  |  avg_loss(CvRMSE) : '+str(round(np.mean(avg_loss)*100,3))+'%'+'  |  avg_time(seconds) : '+str(round(np.mean(avg_time),3))+'\n')
  f.close()
  total_loss.append(np.mean(avg_loss))
  total_time.append(np.mean(avg_time))
f= open('prediction_result.txt','a')
f.write('############# Total average prediction result #############\n')
f.write('total_avg_loss(CvRMSE) : '+str(round(np.mean(total_loss)*100,3))+'%'+'  |  total_avg_time(seconds) : '+str(round(np.mean(total_time),3))+'\n')
f.close()

print("TEST COMPLETE!")
print("Evaluation results are saved in 'prediction_result.txt' file.")
