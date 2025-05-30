import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from .nn_score import nn_score

def build_autoencoder(input_dim):
    inp = Input(shape=(1, input_dim))
    enc = LSTM(64, activation='relu')(inp)
    dec = RepeatVector(1)(enc)
    dec = LSTM(64, activation='relu', return_sequences=True)(dec)
    out = TimeDistributed(Dense(input_dim))(dec)
    autoencoder = Model(inp, out)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

class AEAccHistory(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, threshold):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.thr = threshold
        self.acc_train = []
        self.acc_val = []

    def on_epoch_end(self, epoch, logs=None):
        ftr = nn_score(self.model, self.X_train)
        fvl = nn_score(self.model, self.X_val)
        ytr_pred = (ftr >= self.thr).astype(int)
        yvl_pred = (fvl >= self.thr).astype(int)
        self.acc_train.append(accuracy_score(self.y_train, ytr_pred))
        self.acc_val.append(accuracy_score(self.y_val, yvl_pred))