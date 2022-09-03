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
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
# tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased") 
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 

if torch.cuda.is_available():        
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CPU is exist.")

class FinansRequest(BaseModel):
    complaint: str

class FinansResponse(BaseModel):
    category: str

   
data_dir = "Packet-main/data/"
feature_dir = "Packet-main/feature_map/rpc_packet_raw_log/"
data_file = (data_dir + "rpc_packet_raw_log.csv")
feature_file = (feature_dir + "rpc_packet_test.csv")

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




# # Tokenize all of the sequences and map the tokens to thier IDs.
# input_ids_train = []
# attention_masks_train = []
# for seq in data_text:
#     encoded_dict = tokenizer.encode_plus(
#                         seq,                             # Sequence to encode
#                         add_special_tokens = True,       # Add "[CLS]" and "[SEP]"
#                         max_length = 128,                
#                         padding = "max_length",          # Pad and truncate
#                         truncation=True,                 # Truncate the seq
#                         return_attention_mask = True,    # Construct attn. masks
#                         return_tensors = "pt",           # Return pytorch tensors
#                    )
    
#     # Add the encoded sequences to the list    
#     input_ids_train.append(encoded_dict["input_ids"])
#     # print(input_ids_train)

#     # And its attention mask
#     attention_masks_train.append(encoded_dict["attention_mask"])

# # print(input_ids_train[0])
# input_ids_train = torch.cat(input_ids_train, dim=0)
# # print(input_ids_train[0])
# attention_masks_train = torch.cat(attention_masks_train, dim=0)

# dataset = TensorDataset(input_ids_train, attention_masks_train) 
# dataloader = DataLoader(dataset,sampler = None, batch_size = 64)


# feature = list()
# for batch in dataloader:
#     ####################
#     b_input_ids = batch[0].to(device).long()
#     b_input_mask = batch[1].to(device).long()
#     ####################

#     with torch.no_grad():      
#         # outputs, hidden_states = model(b_input_ids, 
#         #                                 token_type_ids = None, 
#         #                                 attention_mask = b_input_mask, 
#         #                                 output_hidden_states = True)
#         hidden_states = model(b_input_ids, 
#                                 token_type_ids = None, 
#                                 attention_mask = b_input_mask).output_hidden_states.to(device)
#         # print(hidden_states[12])
#         hidden_states = torch.cat(hidden_states, dim = 0) #[13,128,768]

        


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

    # model(b_input_ids, 
    #                          token_type_ids=None, 
    #                          attention_mask=b_input_mask, 
    #                          labels=b_labels)

    # Add the encoded sequences to the list    
    # input_ids_new.append(encoded_dict['input_ids'])
    # And its attention mask
    # attention_masks_new.append(encoded_dict['attention_mask'])

    # print(encoded_dict['input_ids'])
    # input_ids = torch.cat(input_ids_new, dim=0)
    # print(input_ids)
    # attention_masks = torch.cat(encoded_dict['attention_mask'], dim=0)

    model.eval()

    with torch.no_grad():
        # print(model(encoded_dict['input_ids']))
        # outputs, hidden_states  = model(encoded_dict['input_ids'], output_hidden_states = True)
        outputs, hidden_states  = model(encoded_dict['input_ids'], token_type_ids=None, attention_mask=encoded_dict['attention_mask'], output_hidden_states = True)
        # print(hidden_states[12])
        hidden_states = torch.cat(hidden_states, dim = 0) #[13,128,768] #暫時取第一個
        # print(hidden_states[12][0])
    # feature.append(np.append(encoded_dict['input_ids'].numpy(), data_score[i]))
    feature.append(np.append(hidden_states[0][0].numpy(), data_score[i]))
    i = i + 1

(pd.DataFrame((np.array(feature)).squeeze(), columns=range(769))).to_csv(feature_file)




