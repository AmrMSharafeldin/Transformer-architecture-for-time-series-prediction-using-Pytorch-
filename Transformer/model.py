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
import torch.optim as optim
from Transformer.Config import Parameters
from torchsummary import summary



class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoderBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, head_size) 
        self.fc2 = nn.Linear(head_size,input_size) 
        self.multihead_attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.conv1 = nn.Conv1d(input_size, ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(ff_dim, input_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
      # Multi-Head Attention Part
      device = "cuda" if torch.cuda.is_available() else "cpu"
      x = self.fc1(inputs)
      x, _ = self.multihead_attention(x, x, x)
      x = self.fc2(x)
      x = self.dropout(x)
      res = x + inputs
      # Feed Forward Part

      x = self.layer_norm1(res)
      x = x.permute(0, 2, 1)  
      x = F.relu(self.conv1(x))
      x = self.dropout(x)
      x = self.conv2(x)
      x = x.permute(0, 2, 1)
      return x + res


class TransformerModel(nn.Module):
    def __init__(self , head_size, num_heads, ff_dim, num_encoder_blocks, mlp_units, dropout, n_seq_in, n_features, n_seq_out):
        super(TransformerModel, self).__init__()
        self.num_encoder_blocks = num_encoder_blocks
        self.encoder = nn.ModuleList([TransformerEncoderBlock(n_features , head_size, num_heads, ff_dim, dropout) for _ in range(num_encoder_blocks)])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.mlp_layers = nn.ModuleList([nn.Linear(n_seq_in, dim) for dim in mlp_units])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(mlp_units))])
        self.output_layer = nn.Linear(mlp_units[-1], n_seq_out)

    def forward(self, inputs):
        device = "cuda" if torch.cuda.is_available() else "cpu"


        x = inputs
        for encoder in self.encoder:
            x = encoder(x)
        x = self.pooling(x).squeeze(2)
        for linear, dropout in zip(self.mlp_layers, self.dropout_layers):
            x = F.relu(linear(x))
            x = dropout(x)

        x = self.output_layer(x)
        return x
    


head_size = Parameters.get("head_size")
num_heads = Parameters.get("num_heads")
ff_dim = Parameters.get("ff_dim")
num_encoder_block = Parameters.get("num_encoder_block")
mlp_units = Parameters.get("mlp_units")
dropout = Parameters.get("dropout")
n_seq_in = Parameters.get("n_seq_in")
n_features = Parameters.get("n_features")
n_seq_out = Parameters.get("n_seq_out")
lr = Parameters.get("lr")

transformer_model = TransformerModel(  head_size, num_heads, ff_dim , num_encoder_block, mlp_units, dropout, n_seq_in, n_features, n_seq_out)
criterion = nn.L1Loss()
optimizer = optim.Adam(transformer_model.parameters(), lr)


