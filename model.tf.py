from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaout_dfScaler
from urllib.parse import urljoin
from dotenv import load_dotenv
from datetime import *
from bson import ObjectId
import pandas as pd
import numpy as np
import requests
import random
import torch
import json
import os
from dataclasses import dataclass
load_dotenv()

from utils.FlattenData import flatten_input, flatten_output
from utils.SafeDataConverters import safe_to_datetime
from modules.database import Database

import tensorflow as tf
from tensorflow.keras import layers, models


class NeuralNet(models.Model):
    def __init__(self, output_dim: int, hidden_dim: int = 4096, dropout: float = 0.2, last_activation: str = 'softmaout_df') -> None:
        """Initializes the Neural Network model.

        This constructor sets up a neural network model based on the VGG16 architecture, 
        followed by a series of fully connected layers with dropout regularization. 
        The final layer uses the specified activation function.

            input_dim (tuple[int]): The dimensions of the input data.
            output_dim (int): The number of output classes.
            hidden_dim (int): The number of hidden units in the fully connected layers. Defaults to 4096.
            dropout (float): The dropout rate. Defaults to 0.2.
            last_activation (str, optional): The activation function for the final layer. Defaults to 'softmaout_df'.
        """
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.dropout2 = layers.Dropout(dropout)
        self.fc3 = layers.Dense(output_dim, activation=last_activation)

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the neural network.

        This method defines the forward pass of the neural network model.

            inputs (tf.Tensor): The input data.

        Returns:
            tf.Tensor: The output of the neural network.
        """
        out_df = self.fc1(inputs)
        out_df = self.dropout1(out_df)
        out_df = self.fc2(out_df)
        out_df = self.dropout2(out_df)
        out_df = self.fc3(out_df)
        return out_df

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'
    metrics: list[str] = ['accuracy']

class CalendarModel:
    def __init__(self):
        self._db = Database(os.getenv("mongodb_user"), os.getenv("mongodb_password"))
        self.config = TrainingConfig()
    
    @property
    def getTrainingData(self) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
        """Retrieves training data from the database.

        This method fetches the raw data from the database, flattens the input and output data,
        and converts them into pandas DataFrames.

        Returns:
            tuple[list[dict], pd.DataFrame, pd.DataFrame]: 
            - The raw data from the database as a list of dictionaries.
            - The flattened input data as a pandas DataFrame.
            - The flattened output data as a pandas DataFrame.
        """
        flat_output: list[dict] = flatten_output(self._db.to_json())
        flat_input: list[dict] = flatten_input(self._db.to_json())
        return self._db.to_json(), pd.DataFrame(flat_input), pd.DataFrame(flat_output)
    
    
    def train(self):
        _, out_df, inp_df = self.getTrainingData
        out_df = out_df.sort_values(by=['order_id', 'sort_order'], ascending=True).convert_dtypes()
        inp_df = inp_df.sort_values(by=['order_id', 'sort_order'], ascending=True).convert_dtypes()
        
        out_df['sort_order'] = out_df.groupby(['order_id', 'product_id'])['sort_order'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
        inp_df['sort_order'] = inp_df.groupby(['order_id', 'product_id'])['sort_order'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
        out_df = out_df.map(safe_to_datetime)
        inp_df = inp_df.map(safe_to_datetime)
        
        out_df['duration_since_order_created'] = out_df['start_at'] - out_df['order_created_at']
        out_df['duration_since_order_created'] = out_df['duration_since_order_created'].dt.total_seconds() / 60 # convert to minutes
        out_df['time_until_delivery'] = out_df['delivery_date'] - out_df['start_at']
        out_df['time_until_delivery'] = out_df['time_until_delivery'].dt.total_seconds() / 60 # convert to minutes
        
        X = pd.DataFrame(out_df[['task_title', 'sort_order', 'material', 'color']])
        y = pd.DataFrame(out_df[['duration_since_order_created', 'time_until_delivery', 'workspace']])