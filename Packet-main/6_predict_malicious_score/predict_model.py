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


model_dir = "Packet-main/feature_map/ftp_packet_raw_log/model.pt"
data_dir = "Packet-main/feature_map/ftp_packet_raw_log/"
data_file = (data_dir + "ftp_packet_feature_map.csv")
output_file = (data_dir + "ftp_packet_malicious_scores.csv")

test = Net(128 ,150, 25, 6)
test.load_state_dict(torch.load(model_dir))
# test.state_dict()

data = pd.read_csv(data_file, index_col=0)
X, Y = [], []

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
P = X.reshape(-1, X.shape[1]).astype('float32')
P = torch.from_numpy(P)

P = test(P)
_, result = torch.max(P.data, 1)
predict = result
# result = {"feature": X, "malicious_score": predict}
predict = torch.reshape(predict, (len(predict), 1))
result = np.concatenate([X, predict], axis=1)
df = pd.DataFrame(result, columns = range(129))
df.to_csv(output_file)


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


eval_matrix = (pd.crosstab(Y, predict))
print(eval_matrix)

count = 0
for i in range(len(Y)):
    if Y[i] == predict[i]:
        count = count + 1
print("Accuracy : {}".format(count/len(Y)))
