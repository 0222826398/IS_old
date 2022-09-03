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

def classify(complaint):
    target_names = categories
    # target_names = ['hesap','iade', 'iptal','kredi', 'kredi-karti', 'musteri-hizmetleri']

    # Tokenize all of the sequences and map the tokens to thier IDs.
    input_ids_new = []
    attention_masks_new = []

    encoded_dict = tokenizer.encode_plus(
                        complaint,                             # Sequence to encode
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = 128,                
                        padding = 'max_length',          # Pad and truncate
                        truncation=True,                 #Truncate the seq
                        return_attention_mask = True,    # Construct attn. masks
                        return_tensors = 'pt',           # Return pytorch tensors
                    )

    # Add the encoded sequences to the list    
    input_ids_new.append(encoded_dict['input_ids'])

    # And its attention mask
    attention_masks_new.append(encoded_dict['attention_mask'])

    input_ids_new = torch.cat(input_ids_new, dim=0)
    attention_masks_new = torch.cat(attention_masks_new, dim=0)


    # Prediction on test set
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids_new, token_type_ids=None, attention_mask=attention_masks_new)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy() 
        predictions = logits[0].tolist() 

    category_name = target_names[predictions.index(max(predictions))]
    # print("The predicted category is:")
    # print(target_names[predictions.index(max(predictions))])
    
    return(category_name)

   

# Load a trained model and vocabulary that you have fine-tuned
model_dir = 'text_classification-main/model_save/'
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Load input data
data_dir = "text_classification-main/data/"
data_file = (data_dir + "mysql_packet_raw_log.csv")
result_dir = "text_classification-main/result/"
result_file = (result_dir + "mysql_packet_result.csv")
label_dir = "text_classification-main/"
label_file = (label_dir + "label.csv")

print("---Read data---")
data_data = pd.read_csv(data_file)
data_data = data_data.dropna()
data_text = data_data.Packet_payload.values
# data_category = data_data.TTP.values
# data_score = data_data.Malicious_score.values
i = 0

# data = np.unique(data_category)

#categories = np.array(data, dtype = tuple)


###############################
label = pd.read_csv(label_file)
label = label.mysql_packet.values
categories = np.array(label, dtype = tuple)
###############################


print("---Predict---")
TTP = list()


TTP.append(np.append(data_text[0], classify(data_text[0])))
(pd.DataFrame(TTP, columns=range(2))).to_csv(result_file)
TTP = list()

for text in data_text[1:]:
    predict = classify(text)
    #print(predict)

    i = i + 1
    TTP.append(np.append(data_text[i], predict))

    if (i % 5000) == 0:
        (pd.DataFrame(TTP, columns=range(2), index = list(range(i - 4999, i + 1)))).to_csv(result_file, mode = 'a', header = False)
        TTP = list()

(pd.DataFrame(TTP, columns=range(2), index = list(range(i - len(TTP) + 1, i + 1)))).to_csv(result_file, mode = 'a', header = False)
    

