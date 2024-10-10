import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urljoin
from dotenv import load_dotenv
from utils.SafeDataConverters import *
from utils.FlattenData import *
from modules.PremoAPI import *
from datetime import *
from bson import ObjectId
import pandas as pd
import numpy as np
import requests
import random
import torch
import json
import os
load_dotenv()

class NeuralNet(Model):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        
        self.nnet = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_size, activation='softmax')
        ])
        
    def call(self, x):
        return self.nnet(x)
    
class Model:
    def __init__(self):
        # Hyperparameters
        self.batch_size = 32
        self.num_epochs = 1000
        self.learning_rate = 0.001
        self.input_size = 0
        self.output_size = 0
        
        self.model = None
        
    @property
    def getModel(self):
        model = NeuralNet(self.input_size, self.output_size)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, X, y):
        self.input_size = X.shape[1]
        self.output_size = y.shape[1]
        
        self.model = self.getModel
        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.num_epochs, verbose=0, validation_split=0.2)
        
        return self.model, history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)
    
    