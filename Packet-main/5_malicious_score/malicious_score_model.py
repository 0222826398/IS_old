from itertools import count
from tempfile import tempdir
from unittest import result
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
random.seed(5)


class Net(nn.Module):
    def __init__(self,in_size,n_hidden1,n_hidden2,out_size,p=0):

        super(Net,self).__init__()
        self.drop=nn.Dropout(p=p)
        self.linear1=nn.Linear(in_size,n_hidden1)
        nn.init.kaiming_uniform_(self.linear1.weight,nonlinearity='relu')
        self.linear2=nn.Linear(n_hidden1,n_hidden2)
        nn.init.kaiming_uniform_(self.linear2.weight,nonlinearity='relu')
        self.linear3=nn.Linear(n_hidden2,n_hidden2)
        nn.init.kaiming_uniform_(self.linear3.weight,nonlinearity='relu')
        self.linear4=nn.Linear(n_hidden2,out_size)
        
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.drop(x)
        x=F.relu(self.linear2(x))
        x=self.drop(x)
        x=F.relu(self.linear3(x))
        x=self.drop(x)
        x=self.linear4(x)
        return x


class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(X).type(torch.FloatTensor)
        self.y=torch.from_numpy(Y).type(torch.LongTensor)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


data_dir = "Packet-main/feature_map/telnet_packet_raw_log/"
data_file = (data_dir + "telnet_packet_feature_map.csv")
output_dir = "Packet-main/feature_map/telnet_packet_raw_log/model.pt"

data = pd.read_csv(data_file, index_col=0)
X, Y = [], []

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

count = 0
for label in Y:
    # print(label)
    label = str(label)
    split_label = label.split(' ')
    temp = 0
    for i in split_label:
        temp = float(i)
        if float(i) > temp:
            temp = float(i)
    Y[count] = np.float64(int(temp))
    # print(Y[count])
    count = count + 1
    
Y = Y.astype(float)
print(X.shape)
print(Y.shape)

data_set=Data()
trainloader=DataLoader(dataset=data_set,batch_size=64)

model=Net(128 ,150, 25, 6)
# model_drop=Net(129, 150, 30, 5, p = 1e-1)
model.train()

optimizer_ofit = torch.optim.Adam(model.parameters(), lr=5e-4)
# optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
LOSS={}
LOSS['training data no dropout']=[]
LOSS['validation data no dropout']=[]
LOSS['training data dropout']=[]
LOSS['validation data dropout']=[]
n_epochs=50

for epoch in range(n_epochs):
    for x, y in trainloader:
        #make a prediction for both models 
        yhat = model(data_set.x)
        # print(yhat)
        # yhat_drop = model_drop(data_set.x)
        # print(yhat_drop)
        #calculate the lossf or both models 
        loss = criterion(yhat, data_set.y)
        # loss_drop = criterion(yhat_drop, data_set.y)

        #store the loss for  both the training and validation  data for both models 
        LOSS['training data no dropout'].append(loss.item())
        # LOSS['training data dropout'].append(loss_drop.item())
        # model_drop.eval()
        # model_drop.train()

        #clear gradient 
        optimizer_ofit.zero_grad()
        # optimizer_drop.zero_grad()
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        # loss_drop.backward()
        #the step function on an Optimizer makes an update to its parameters
        optimizer_ofit.step()
        # optimizer_drop.step()
        
        print('epoch {}, loss {}'.format(epoch, loss.item()))


torch.save(model.state_dict(), output_dir)


result = model(data_set.x)
_,yhat=torch.max(result.data,1)
eval_matrix = (pd.crosstab(Y, yhat))
print(eval_matrix)

