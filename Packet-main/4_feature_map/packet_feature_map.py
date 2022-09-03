#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertForSequenceClassification, 
                          BertTokenizer, 
                          AdamW, 
                          BertConfig, 
                          get_linear_schedule_with_warmup,
                         )
from typing import Optional
from pydantic import BaseModel
import pandas as pd

output_dir = 'Packet-main/model_save/'
# model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased") 
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 


class FinansRequest(BaseModel):
    complaint: str

class FinansResponse(BaseModel):
    category: str

   
data_dir = "Packet-main/data/"
feature_dir = "Packet-main/feature_map/rpc_packet_raw_log/"
data_file = (data_dir + "rpc_packet_raw_log.csv")
feature_file = (feature_dir + "rpc_packet_feature_map3.csv")

print("---Read data---")
#-----------------train file-------------------------
data_data = pd.read_csv(data_file)
data_data = data_data.dropna()
data_text = data_data.Packet_payload.values
data_category = data_data.TTP.values
data_score = data_data.Malicious_score.values
i = 0

train = np.unique(data_category)
print("---Complete reading data---")
categories = np.array(train, dtype = tuple)

feature = list()
for text in data_text:
    target_names = categories


    # Tokenize all of the sequences and map the tokens to thier IDs.
    input_ids_new = []
    attention_masks_new = []

    encoded_dict = tokenizer.encode_plus(
                        text,                             # Sequence to encode
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = 128,                
                        padding = 'max_length',          # Pad and truncate
                        truncation=True,                 #Truncate the seq
                        return_attention_mask = True,    # Construct attn. masks
                        return_tensors = 'pt',           # Return pytorch tensors
                    )


    feature.append(np.append(encoded_dict['input_ids'].numpy(), data_score[i]))
    i = i + 1

(pd.DataFrame((np.array(feature)).squeeze(), columns=range(129))).to_csv(feature_file)




