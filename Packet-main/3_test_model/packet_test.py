#!/usr/bin/env python
# coding: utf-8
###因為資料不均勻的關係test_data編碼出來的結果跟train不一樣造成預測錯誤
###此bug利用41、48、61、62

# encoding=utf-8
from re import X
from lightgbm import train
import pandas as pd
import numpy as np
import time
import csv
import os
import random
import datetime
import operator
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertForSequenceClassification, 
                          BertTokenizer, 
                          AdamW, 
                          BertConfig, 
                          get_linear_schedule_with_warmup,
                         )
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Check the available GPU and use it if it is exist. Otherwise use CPU
if torch.cuda.is_available():        
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CPU is exist.')
    
    
# Read Dataset    
data_dir = "Packet-main/model/telnet_packet_raw_log/"
output_dir = data_dir
# output_dir = "Packet-main/model/fulldata_telnet_packet/"

###這邊有玉庭式更改
train_file = (data_dir + 'train.csv')
test_file = (data_dir + 'test.csv')
###這邊有玉庭式更改

print("---Read data---")
#-----------------test file-------------------------
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

###這邊有玉庭式更改
test_text = test_data.Packet_payload.values
test_category = test_data.TTP.values
###這邊有玉庭式更改

print("---Complete reading data---")

# Convert non-numeric labels to numeric labels. 
###
###這邊有玉庭式更改
train = np.unique(train_data.TTP.values)
test = np.unique(test_category)
train = np.unique(np.append(train, test))

categories = np.array(train, dtype = tuple)

# test = np.unique(test_category)
# categories = np.array(test, dtype = tuple)
###這邊有玉庭式更改
le = preprocessing.LabelEncoder()

def numeric_category(test_category):
    categories_df = pd.DataFrame(categories, columns=['category'])
    categories_df['labels'] = le.fit_transform(categories_df['category'])
    
    ###這邊有玉庭式更改
    le.fit(categories_df["category"])
    ###這邊有玉庭式更改
    test_labels = le.transform(test_category)
    return test_labels, categories_df

test_labels, categories_df = numeric_category(test_category)

print('\n')
print("Convert non-numeric labels to numeric labels\n")
print(categories_df.sort_values(by='category', ascending=True))


# Load the trained model 
# output_dir = 'text_classification-main/model_save/'

model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

model.to(device)


# Tokenize all of the sequences and map the tokens to thier IDs.
input_ids_test = []
attention_masks_test = []

# For every sequences
for seq in test_text:
    encoded_dict = tokenizer.encode_plus(
                        seq,                             # Sequence to encode
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = 128,                
                        padding = 'max_length',          # Pad and truncate
                        truncation=True,                 #Truncate the seq
                        return_attention_mask = True,    # Construct attn. masks
                        return_tensors = 'pt',           # Return pytorch tensors
                   )
    
    # Add the encoded sequences to the list    
    input_ids_test.append(encoded_dict['input_ids'])

    # And its attention mask
    attention_masks_test.append(encoded_dict['attention_mask'])
    
input_ids_test = torch.cat(input_ids_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)
test_labels = torch.tensor(test_labels)

#Specify batch size
batch_size = 32  

prediction_data = TensorDataset(input_ids_test, attention_masks_test, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# Prediction on test set
print('\n')
print('Predicting labels for {:,} test sentences'.format(len(input_ids_test)))

model.eval()
predictions , true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()    
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)
    
    
#Evaluation the result
true_labels_array, result_label, result_prob, result_logits = [], [], [], []
for j in range(len(true_labels)):
    for i in range(len(true_labels[j])):
        true_labels_array.append(true_labels[j][i])


for j in range(len(predictions)):
    for i in range(len(predictions[j])):      
        index, value = max(enumerate(predictions[j][i]), key=operator.itemgetter(1))
        result_label.append(index)
        result_prob.append(value)
        result_logits.append(predictions[j][i])
        
###這邊有玉庭式更改
target_names = categories
target_name = sorted(list(set(true_labels_array)))
target_names = list()
for i in target_name:
    target_names.append(categories_df.category[i])
# target_names = ['hesap','iade', 'iptal','kredi', 'kredi-karti', 'musteri-hizmetleri']
###這邊有玉庭式更改

print("Accuracy     ", accuracy_score(test_labels, result_label))
print("Precision    ", precision_score(test_labels, result_label, average="macro"))
print("Recall       ", recall_score(test_labels, result_label, average='macro'))
print("F1           ", f1_score(test_labels, result_label, average="macro"))
print(true_labels_array)
print(result_label)
# print(classification_report(true_labels_array, result_label, target_names = target_names))