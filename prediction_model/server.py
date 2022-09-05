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

DATA_INDEX = int(sys.argv[1])
  
torch.manual_seed(123)

# Convert data to form of sequence data
def seq_data(x, sequence_length):
    
    x_seq = []
    y_seq = []
    for i in range(len(x)-sequence_length-1):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(x[i+sequence_length])
    
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1]) 

# Read data
with open('/home/proj01/gangmin/test/datas.pkl','rb') as f:
    data = pickle.load(f)
columns = data.columns
X = data[columns[DATA_INDEX]].values
data_name = columns[DATA_INDEX]

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

# Generate Data Loader
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)


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
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = math.sqrt(loss)/torch.mean(outputs)
            running_loss += loss.item()
        running_loss_arr.append(running_loss/n)
        
    model.train()
    return running_loss_arr

# LSTM Model Config
input_size = x_seq.size(2)
num_layers = 1
hidden_size = 8

# Initialize the model
model = LSTM(input_size=input_size, 
                   hidden_size=hidden_size, 
                   sequence_length=sequence_length, 
                   num_layers=num_layers, device=device)    
model = model.to(device)

# Global model evaluate 
def get_eval_fn(model):
        # model with evaluation
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float,float]]:
        params_dict = zip(model.state_dict().keys(),weights)
        state_dict = OrderedDict({k:torch.tensor(v) for k,v in params_dict})
        model.load_state_dict(state_dict,strict=True)
        
        with open('weight_'+data_name+'.pkl','wb') as f:
          pickle.dump(model,f)
        
        f= open('time_log.txt','a')
        f.write("Model.pkl export time for "+data_name+" : "+str(time.strftime('%X', time.localtime()))+"\n")
        f.close()
          
        test_loss = accuracy(test_loader, model) # 시험 데이터의 Accuracy
        print("##############",test_loss)
        
        return 0, {"err":0}
    return evaluate


# Define strategy
strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_available_clients=5,
        min_fit_clients =5,
        min_eval_clients = 1,
        eval_fn = get_eval_fn(model)
        )

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 10},
    strategy=strategy,
)
f= open('time_log.txt','a')
f.write("Server start time for "+data_name+ " : "+str(time.strftime('%X', time.localtime()))+"\n")
f.close()

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

f= open('prediction_result.txt','w')
f.write('############# prediction result #############\n')
f.write('data : '+'%15s'%data_name)
f.write('  |  avg_loss : '+str(np.mean(test_loss))+'  |  avg_time : '+str(np.mean(avg_time))+'\n')
f.close()