import torch.optim as optim  
import warnings
from collections import OrderedDict
import pickle
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math


import sys

DATA_INDEX = int(sys.argv[1]) 
device = torch.device("cpu")

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

split = int(len(X)*0.7)
sequence_length = 8
X = np.expand_dims(X, axis=1)

x_seq, y_seq = seq_data(X, sequence_length)
# x_seq = x_seq.unsqueeze(axis=2)

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

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cpu")

# LSTM Model Config
input_size = x_seq.size(2)
num_layers = 1
hidden_size = 8

# Model
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

def train(model, trainloader):
    """Train the model on the training set."""
    loss_graph = []
    n = len(train_loader)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(5):
        running_loss = 0.0
      
        for data in train_loader:
            seq, target = data # Mini-batch
            out = model.forward(seq)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
      
        loss_graph.append(running_loss/n)
        print('[epoch: %d] loss: %.4f' %(epoch, running_loss/n))


# Initialize the model
model = LSTM(input_size = input_size, 
                   hidden_size = hidden_size, 
                   sequence_length = sequence_length, 
                   num_layers = num_layers, device = device)    

model = model.to(device) 

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, train_loader)
        
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        
        return 0, 0, {"accuracy": 0}

# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())