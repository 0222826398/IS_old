#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import os
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import word_tokenize

#add "tl" to nltk"s stopword list in turkish
all_stopwords = stopwords.words("turkish")
all_stopwords.append("tl")
nltk.download("punkt")


def normalize_text(text):

    #do lowercase 
    result_lower = "".join((text.lower()))
    #remove numbers
    # result_non_numeric = "".join([i for i in result_lower if not i.isdigit()])
    #remove punctuations
    # result_non_punc = re.sub(r"[*"`,”+;,_,’^#@<=>~&$€%[:!\-\"\\\/}{?\,.()[\]{|}]","",result_non_numeric).strip()
    #remove whitespace
    # result_non_space = re.sub(" +"," ",result_non_punc)
    
    #remove stopwords
    # text_tokens = word_tokenize(result_non_space)
    # tokens_without_stopword = [word for word in text_tokens if not word in all_stopwords]
    # filtered_text = (" ").join(tokens_without_stopword)
    
    return (result_lower)



# read the dataset and then split into train and test data
filename = ("text_classification-main/data/ftp_session_raw_log.csv")
data = pd.read_csv(filename)
# print(data)
# data = data.drop(["index"], axis=1)
# data = data.drop(["Malicious_score"], axis=1)
# print(data)
#=================================Split train/test/dev ============================
xTrain, xTest = train_test_split(data, test_size = 0.1, random_state = 0)

#--------------------write train/test/dev---------------------------------

def write (data, path):
    print("---Writing starts---") 
    text = data.Session_payload.values
    
    filtered_text = []
    for seq in text:
        filtered_text.append(normalize_text(seq))
        
    category = data.TTP.values
    row_data = {"Session_payload":filtered_text, "TTP":category}
    df = pd.DataFrame(row_data, columns = ["Session_payload", "TTP"])
    df.to_csv(path)
    print("---Writing ends---") 
    return 

# Write the files if there are not exist 
if not os.path.exists("text_classification-main/data/train.csv" and "text_classification-main/data/test.csv"):
    writeFileTrain = ("text_classification-main/data/train.csv")
    writeFileTest = ("text_classification-main/data/test.csv")
    write(xTrain, writeFileTrain)
    write(xTest, writeFileTest)

