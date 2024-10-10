import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime, timedelta

# Custom loss function to handle constraints
def custom_loss(penalty_factor):
    def loss(y_true, y_pred):
        # Mean Squared Error for regression targets
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Extract predictions
        date_offset_pred = y_pred[:, 0]
        start_time_pred = y_pred[:, 1]
        
        # Constraint violations
        time_penalty = tf.reduce_mean(
            tf.maximum(0.0, 9 - start_time_pred) + tf.maximum(0.0, start_time_pred - 17)
        )
        # Penalize date_offset less than 1 (today or past)
        date_penalty = tf.reduce_mean(tf.maximum(0.0, 1 - date_offset_pred))
        
        # Total loss
        total_loss = mse + (time_penalty + date_penalty) * penalty_factor
        return total_loss
    return loss

# Neural network model
class NeuralNet(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        
        self.nnet = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_size)  # Output layer for date offset and start time
        ])
        
    def call(self, x):
        return self.nnet(x)
    
# Model management class
class Model:
    def __init__(self):
        # Hyperparameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.input_size = 0
        self.output_size = 2  # Predicting date offset and start time
        self.penalty_factor = 10.0  # Adjust as needed
        
        self.model = None
        
    def getModel(self):
        model = NeuralNet(self.input_size, self.output_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=custom_loss(self.penalty_factor))
        return model
    
    def train(self, X, y):
        self.input_size = X.shape[1]
        
        self.model = self.getModel()
        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=1,
            validation_split=0.2
        )
        
        return self.model, history
    
    def adjust_predictions(self, predictions, order_created_at_list, task_duration_list):
        adjusted_predictions = []
        for idx, pred in enumerate(predictions):
            date_offset, start_time = pred
            order_created_at = order_created_at_list[idx]
            
            # Ensure start time is within [9, 17]
            start_time = np.clip(start_time, 9, 17)
    
            # Ensure date offset is at least 1 (tomorrow or later)
            date_offset = max(1, np.ceil(date_offset))
    
            # Calculate the proposed start date and time
            proposed_date = order_created_at + timedelta(days=date_offset)
    
            # If the proposed date is on a weekend, move it to the next Monday
            while proposed_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                proposed_date += timedelta(days=1)
    
            # Combine date and time into a datetime object
            start_at = proposed_date.replace(
                hour=int(start_time),
                minute=int((start_time % 1) * 60),
                second=0,
                microsecond=0
            )
            end_at = start_at + timedelta(hours=task_duration_list[idx])  # Assuming task_duration_list is available
    
            adjusted_predictions.append((start_at, end_at))
    
        return adjusted_predictions
        
    def predict(self, X: np.ndarray, order_created_at_list, task_duration_list) -> np.ndarray:
        predictions = self.model.predict(X, verbose=0)
        adjusted_predictions = self.adjust_predictions(predictions, order_created_at_list, task_duration_list)
        return adjusted_predictions
