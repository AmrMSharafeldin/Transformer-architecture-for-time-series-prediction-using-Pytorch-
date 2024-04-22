import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.optim as opti

from Transformer.model import TransformerEncoderBlock, TransformerModel , transformer_model, criterion, optimizer
from Transformer.utils import *
from Transformer.Config import Parameters

import os 
epochs = Parameters.get("epochs")
batch_size = Parameters.get("batch_size")
val_size = Parameters.get("val_size")

head_size = Parameters.get("head_size")
num_heads = Parameters.get("num_heads")
ff_dim = Parameters.get("ff_dim")
num_encoder_block = Parameters.get("num_encoder_block")
mlp_units = Parameters.get("mlp_units")
dropout = Parameters.get("dropout")
n_seq_in = Parameters.get("n_seq_in")
n_features = Parameters.get("n_features")
n_seq_out = Parameters.get("n_seq_out")
features = Parameters.get("features")

IBM_path = 'data/train_data/IBM.csv'
GOOGL_path = 'data/train_data/MSFT.csv'
MSFT_path = 'data/train_data/GOOG.csv'
lr = Parameters.get("lr")


def train(model,train_data):
    train_data.normalzie_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainX_tensor = torch.tensor(train_data.X_train, dtype=torch.float32).to(device)
    trainY_tensor = torch.tensor(train_data.y_train, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train() 
    total_loss = 0.0
    start_time = time.time()
    for epoch in range(epochs):
      total_loss  , start_time = 0.0 , time.time()
      for batch_X, batch_Y in train_loader:
          optimizer.zero_grad()
          output = model(batch_X)
          loss = criterion(output, batch_Y)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
          elapsed = time.time() - start_time
      print(f"Epoch {epoch+1}, Estimated Time : {elapsed} Average Loss: {total_loss/(batch_size)}")
    return model


def conctenate_data():

    
    IBM_data = pd.read_csv(IBM_path, delimiter=',', usecols=features)
    GOOGL_data = pd.read_csv(GOOGL_path, delimiter=',', usecols=features)
    MSFT_data = pd.read_csv(MSFT_path, delimiter=',', usecols=features)
    data_class_imb  = data_extractor(IBM_data,val_size,n_seq_in ,n_features , n_seq_out  ,features )
    data_class_google  = data_extractor(GOOGL_data,val_size,n_seq_in ,n_features , n_seq_out  ,features )
    data_class_msft  = data_extractor(MSFT_data,val_size,n_seq_in ,n_features , n_seq_out  ,features )
    data_class_imb.stack_data(data_class_google.X_train ,data_class_google.y_train )
    data_class_imb.stack_data(data_class_msft.X_train ,data_class_msft.y_train )
    return data_class_imb


def train_transformer():
    data= conctenate_data() 
    model = train(transformer_model , data)
    save_model(model, Parameters.get("model_save_path"))
