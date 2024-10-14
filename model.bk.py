import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime, timedelta
import os

# Custom loss function to handle constraints
def custom_loss(penalty_factor):
    def loss(y_true, y_pred):
        # Mean Squared Error for regression targets
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Extract predictions
        date_offset_pred = y_pred[:, 0]
        
        # Penalize date_offset less than 1 (today or past)
        # date_penalty = tf.reduce_mean(tf.maximum(0.0, 1 - date_offset_pred))
        
        # Total loss
        total_loss = mse # + date_penalty * penalty_factor
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
        self.penalty_factor = 1.0  # Adjust as needed
        self.model_path = None
        self.model = None
        
    def getModel(self):
        model = NeuralNet(self.input_size, self.output_size)
        if self.model_path:
            latest_checkpoint = tf.train.latest_checkpoint(self.model_path)
            if latest_checkpoint:
                model.load_weights(latest_checkpoint)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=custom_loss(self.penalty_factor))
        return model
    
    def __calback(self):
        if not self.model_path: return None
        if not os.path.isdir(self.model_path): os.makedirs(self.model_path)
        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        
        return tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, f'{date}.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0
        )
        
    
    def train(self, X_train, y_train, X_val, y_val):
        self.input_size = X_train.shape[1]
        self.model = self.getModel()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[self.__calback()]
        )
        
        return self.model, history
    
    def adjust_prediction(self, predictions, order_created_at_list: list[datetime], task_duration_list: list[float|int], product_operation_id_list: list[int], scaler=None) -> np.ndarray:
        adjusted_pred: list[dict] = []
        
        for idx, pred in enumerate(predictions):
            date_offset, start_time = pred
            order_created_at = order_created_at_list[idx].astype('M8[ms]')
            task_duration = task_duration_list[idx]
            product_operation_id = product_operation_id_list[idx]
            
            if scaler:
                date_offset = scaler.inverse_transform(np.array([[date_offset, 0]]))[0][0]
                start_time = scaler.inverse_transform(np.array([[0, start_time]]))[0][1]
            
            start_at = order_created_at.astype(datetime) + timedelta(days=date_offset)
            start_at = start_at.replace(hour=0, minute=0, second=0, microsecond=0)
            start_at += timedelta(hours=start_time)
            start_at = start_at.replace(second=0, microsecond=0)
            
            # Ensure start time is not before 09:00
            if start_at.time() < datetime.strptime("09:00", "%H:%M").time():
                start_at = start_at.replace(hour=9, minute=0, second=0, microsecond=0)
            
            # Calculate end time
            end_at = start_at + timedelta(seconds=int(task_duration))
            
            # Split tasks if they go beyond business hours or span multiple days
            while start_at.date() != end_at.date() or end_at.time() > datetime.strptime("17:00", "%H:%M").time():
                # Calculate the end of the current business day
                end_of_day = start_at.replace(hour=17, minute=0, second=0, microsecond=0)
                
                # If the task spans multiple days, split it
                if end_at.date() != start_at.date() or end_at.time() > datetime.strptime("17:00", "%H:%M").time():
                    task_duration_today = (end_of_day - start_at).total_seconds()
                    task_card = {
                        'product_operation_id': product_operation_id,
                        'task_duration': task_duration_today,
                        'start_at': start_at,
                        'end_at': end_of_day,
                        'order_created_at': order_created_at_list[idx]
                    }
                    adjusted_pred.append(task_card)
                    
                    # Move to the next business day
                    start_at = start_at + timedelta(days=1)
                    start_at = start_at.replace(hour=9, minute=0, second=0, microsecond=0)
                    task_duration -= task_duration_today
                    end_at = start_at + timedelta(seconds=int(task_duration))
                else:
                    break
            
            # Ensure the final task card is not on a weekend
            while start_at.weekday() > 5:
                start_at = start_at + timedelta(days=1)
                start_at = start_at.replace(hour=9, minute=0, second=0, microsecond=0)
                end_at = start_at + timedelta(seconds=int(task_duration))
            
            # Add the final task card
            task_card = {
                'product_operation_id': product_operation_id,
                'task_duration': task_duration,
                'start_at': start_at,
                'end_at': end_at,
                'order_created_at': order_created_at_list[idx]
            }
            adjusted_pred.append(task_card)
            
            
        for tk in adjusted_pred:
            if tk['start_at'].weekday() >= 5:
                tk['start_at'] = tk['start_at'] + timedelta(days=(7 - tk['start_at'].weekday()))
                tk['start_at'] = tk['start_at'].replace(hour=9, minute=0, second=0, microsecond=0)
                tk['end_at'] = tk['start_at'] + timedelta(seconds=int(tk['task_duration']))
            
        return adjusted_pred
        
    def predict(self, X: np.ndarray, order_created_at_list: list[datetime], task_duration_list: list[float|int], product_operation_id_list:list[int], scaler=None) -> np.ndarray:
        predictions = self.model.predict(X, verbose=0)
        adjusted_predictions = self.adjust_prediction(predictions, order_created_at_list, task_duration_list, product_operation_id_list, scaler)
        return adjusted_predictions
