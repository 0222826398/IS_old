#!/usr/bin/env python
# coding: utf-8
###這邊有玉庭式更改
from cgi import test
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
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


total_t0 = time.time()
# Check the available GPU and use it if it is exist. Otherwise use CPU
if torch.cuda.is_available():        
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CPU is exist.")
    

# Read the Dataset
data_dir = "text_classification-main/data/"

data = (data_dir + "telnet_session_raw_log.csv")

print("---Read data---")
#-----------------train file-------------------------
train_data = pd.read_csv(data)
###這邊有玉庭式更改
train_data = train_data.dropna()
train_text = train_data.Session_payload.values
train_category = train_data.TTP.values
###這邊有玉庭式更改


# Convert non-numeric labels to numeric labels. 
###這邊有玉庭式更改
# categories = ("hesap","iade", "iptal","kredi", "kredi-karti", "musteri-hizmetleri")
train = np.unique(train_category)
print("---Complete reading data---")
categories = np.array(train, dtype = tuple)
###這邊有玉庭式更改
le = preprocessing.LabelEncoder()

def numeric_category(train_category):
    categories_df = pd.DataFrame(categories, columns=["category"])
    categories_df["labels"] = le.fit_transform(categories_df["category"])
    # print(categories_df)

    ###這邊有玉庭式更改
    le.fit(categories_df["category"])
    # le.fit(test_category)
    ###這邊有玉庭式更改
    train_labels = le.transform(train_category)
    return train_labels, categories_df

train_labels, categories_df = numeric_category(train_category)

###這邊有玉庭式更改
label_num = int(train_labels.max()) + 1
###這邊有玉庭式更改

print("Convert non-numeric labels to numeric labels\n")
print(categories_df.sort_values(by="category", ascending=True))

#Text Tokenization
#Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")   

#Specify maximum sequence length to pad or truncate
max_len = 0

c = 0

for seq in train_text:
    # Tokenize the text by BERT tokenizer
    ###這邊有玉庭式更改
    # input_ids = tokenizer.encode(seq, add_special_tokens=True)
    # print(seq)
    input_ids = tokenizer.encode(seq, add_special_tokens=True, truncation=True, max_length = 512)
    c = c + 1
    ###這邊有玉庭式更改
    max_len = max(max_len, len(input_ids))

print("Maximum sequence length", max_len)


# Tokenize all of the sequences and map the tokens to thier IDs.
input_ids_train = []
attention_masks_train = []

# For every sequences
for seq in train_text:
    encoded_dict = tokenizer.encode_plus(
                        seq,                             # Sequence to encode
                        add_special_tokens = True,       # Add "[CLS]" and "[SEP]"
                        max_length = 128,                
                        padding = "max_length",          # Pad and truncate
                        truncation=True,                 #Truncate the seq
                        return_attention_mask = True,    # Construct attn. masks
                        return_tensors = "pt",           # Return pytorch tensors
                   )
    
    # Add the encoded sequences to the list    
    input_ids_train.append(encoded_dict["input_ids"])

    # And its attention mask
    attention_masks_train.append(encoded_dict["attention_mask"])
    
input_ids_train = torch.cat(input_ids_train, dim=0)
attention_masks_train = torch.cat(attention_masks_train, dim=0)
train_labels = torch.tensor(train_labels)


# Change to TensorDataset and Split to train and validation sets (90-10)
dataset = TensorDataset(input_ids_train, attention_masks_train, train_labels) 
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset = dataset

# print("training set", format(train_size))
# print("validation set", format(val_size))


#specify batch size
batch_size = 32

train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset), batch_size = batch_size)
# validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

#Specify Classification model

model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased", 
    ###這邊有玉庭式更改
    num_labels = label_num,    
    ###這邊有玉庭式更改             
    output_attentions = False, 
    output_hidden_states = False,
)

# Run the model on GPU
# uncomment this if use GPU
model.cuda()

#Specify the optimizer and epoch number
optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)

epochs = 4    # recomende 2-4 by BERT model"s authors
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Traing start
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#use training_status to store loss values, accuracy and elapsed time
training_status = []
# total_t0 = time.time()

for epoch_i in range(0, epochs):

    #-------------------Training-----------------------
    print("Epoch {:} / {:}".format(epoch_i + 1, epochs))

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 200 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

        ###這邊有玉庭式更改
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()
        ###這邊有玉庭式更改

        model.zero_grad()       

        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)             
    training_time = format_time(time.time() - t0)

    print("\n")
    print(" Average training loss: {0:.2f}".format(avg_train_loss))
    print(" Training epcoh took: {:}".format(training_time))
         
    # ------------------Validation--------------------
    # print("\n")
    # print("Validation")

    # t0 = time.time()
    # model.eval()

    # total_eval_accuracy = 0
    # total_eval_loss = 0
    # nb_eval_steps = 0

    # for batch in validation_dataloader:
        ###這邊有玉庭式更改
    #     b_input_ids = batch[0].to(device).long()
    #     b_input_mask = batch[1].to(device).long()
    #     b_labels = batch[2].to(device).long()
    #     ###這邊有玉庭式更改

    #     with torch.no_grad():      
    #         (loss, logits) = model(b_input_ids, 
    #                                token_type_ids=None, 
    #                                attention_mask=b_input_mask,
    #                                labels=b_labels)


    #     total_eval_loss += loss.item()

    #     logits = logits.detach().cpu().numpy()
    #     label_ids = b_labels.to("cpu").numpy()

    #     total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # validation_time = format_time(time.time() - t0)
    
    # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print("  Validation took: {:}".format(validation_time))

    training_status.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            # "Valid. Loss": avg_val_loss,
            # "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            # "Validation Time": validation_time
        }
    )

print("\n")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


output_dir = "text_classification-main/model/fulldata_telnet_session/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, "module") else model 
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

