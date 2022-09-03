from cgi import test
import pandas as pd
import numpy as np
import time
import csv
import os


data_dir = "text_classification-main/data/"
data = (data_dir + "mysql_packet_raw_log.csv")

label_dir = "text_classification-main/"
label = (label_dir + "label.csv")

print("---Read data---")
#-----------------train file-------------------------
train_data = pd.read_csv(data)
###這邊有玉庭式更改
train_data = train_data.dropna()
train_category = train_data.TTP.values
###這邊有玉庭式更改


# Convert non-numeric labels to numeric labels. 
###這邊有玉庭式更改
# categories = ("hesap","iade", "iptal","kredi", "kredi-karti", "musteri-hizmetleri")
train = np.unique(train_category)
print("---Complete reading data---")
categories = np.array(train, dtype = tuple)
# (pd.DataFrame((np.array(feature)).squeeze(), columns=range(129))).to_csv(feature_file)
(pd.DataFrame(categories)).to_csv(label)