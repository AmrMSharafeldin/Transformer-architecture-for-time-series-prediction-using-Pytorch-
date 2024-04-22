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

class data_extractor:
  def __init__(self, data,val_size, n_seq_in, n_features , n_seq_out, features = ["Close"]):
    self.data = data
    self.val_size = val_size
    self.n_seq_in = n_seq_in
    self.n_seq_out = n_seq_out
    self.features = features
    self.n_features = n_features
    self.train, self.eval = self.data_loader()
    self.X_train, self.y_train = self.sliding_window(self.train)
    self.X_eval, self.y_eval = self.sliding_window(self.eval)
    self.scaler_y = StandardScaler()
  def feature_extractor(self) -> pd.Series:
    return self.data[self.features]
  
  
  def split_data(self)->tuple:
    data = self.feature_extractor()
    if len(data) != 0:
        split_idx = round(len(data) * (1-self.val_size))
        train = data[:split_idx]
        test = data[split_idx:]
        train = np.array(train)
        test = np.array(test)
        #print(train[:, np.newaxis].shape)
        return train[:, np.newaxis], test[:, np.newaxis]
    else:
        raise Exception('Data set is empty!!!')
    
  def reshape(self, data) -> np.array:
    buffer = np.array(np.array_split(data, int(data.shape[0] / self.n_seq_in)))
    buffer.reshape((int(data.shape[0] / self.n_seq_in), self.n_seq_in, self.n_features))
    data = buffer.reshape((buffer.shape[0]*buffer.shape[1], buffer.shape[2]))
    return data 
  def data_loader(self)->tuple[np.array, np.array]:
    train , test = self.split_data()
    train_index = train.shape[0] % self.n_seq_in
    test_index = test.shape[0] % self.n_seq_in
    train = train[train_index:] if train_index != 0 else train
    test = test[test_index:] if test_index != 0 else test
    return self.reshape(train), self.reshape(test)

  def sliding_window(self , data)->tuple:
        X, y = [], []
        startindex = 0
        for _ in range(len(data)):
            endIndex = startindex + self.n_seq_in
            out_end = endIndex + self.n_seq_out
            if out_end <= len(data):
                x_input = data[startindex:endIndex, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[endIndex:out_end, 0])
                startindex += 1
        return np.array(X), np.array(y)
    
  def normalzie_data(self):
    scaler_x = StandardScaler()
    trainX , trainY,evalX , evalY = self.X_train, self.y_train , self.X_eval, self.y_eval
    self.X_train, self.y_train = scaler_x.fit_transform(trainX.reshape(-1, trainX.shape[-1])).reshape(trainX.shape) , self.scaler_y.fit_transform(trainY.reshape(-1, trainY.shape[-1])).reshape(trainY.shape)
    self.X_eval, self.y_eval = scaler_x.fit_transform(evalX.reshape(-1, evalX.shape[-1])).reshape(evalX.shape) , self.scaler_y.fit_transform(evalY.reshape(-1, evalY.shape[-1])).reshape(evalY.shape)
  def denormalzie_evaldata(self , eval_data):
   return self.scaler_y.inverse_transform(eval_data.detach().numpy().reshape(-1, eval_data.shape[-1])).reshape(eval_data.shape)
  
  def stack_data(self , x1, y1):
    self.X_train = np.vstack((self.X_train, x1))
    self.y_train = np.vstack((self.y_train, y1))


def plot_predictions(plot_test, plot_preds, shape, show = False):
    fig, ax = plt.subplots(figsize=(20,6))
    
    # Create x-axis range based on the shape of the input data
    x = shape

    # Plotting
    ax.plot(x, plot_test, label='actual')
    ax.plot(x, plot_preds, label='preds')
    ax.set_title('Predictions vs. Actual')
    ax.set_xlabel('Index')
    ax.set_ylabel('Stock Price')
    ax.legend()
    if(show):plt.show()
    return fig
def save_model(model, save_path):
    if save_path is not None:
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        torch.save(model.state_dict(), save_path)

def load_model(model, path):
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
    model.load_state_dict(torch.load(path))