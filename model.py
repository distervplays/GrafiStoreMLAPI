import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os

# Custom loss function to handle constraints
def custom_loss():
    """ Custom loss function to penalize date offsets less than 1
    
    Args:
        penalty_factor (float): The penalty factor for date offsets less than 1
        
    Returns:
        loss (function): The custom loss function
    """
    def loss(y_true, y_pred):
        """ Custom loss function to penalize date offsets less than 1
        
        Args:
            y_true (tf.Tensor): The true values
            y_pred (tf.Tensor): The predicted values
            
        Returns:
            total_loss (tf.Tensor): The total loss
        """
        # Mean Squared Error for regression targets
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        
        total_loss = mse # Total loss
        return total_loss
    return loss

# Neural network model
class NeuralNet(tf.keras.Model):
    def __init__(self, input_size, output_size):
        """ Neural network model for predicting date offset and start time
        
        Args:
            input_size (int): The number of input features
            output_size (int): The number of output features
            
        Returns:
            None
        """
        
        super(NeuralNet, self).__init__()
        
        self.nnet = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_size)  # Output layer for date offset and start time
        ])
        
    def call(self, x):
        """ Forward pass of the neural network

        Args:
            x (tf.Tensor): The input tensor
            
        Returns:
            y (tf.Tensor): The output tensor
        """
        return self.nnet(x)
    
# Model management class
class Model:
    def __init__(self) -> None:
        """
        Initializes the model with default hyperparameters and attributes.
        This method sets up the initial configuration for the model, including hyperparameters such as batch size, 
        number of epochs, learning rate, input size, and output size. It also initializes placeholders for the model 
        path and the model instance itself.
        
        Attributes:
            batch_size (int): The number of samples per gradient update.
            num_epochs (int): The number of epochs to train the model.
            learning_rate (float): The learning rate for the optimizer.
            input_size (int): The number of input features.
            output_size (int): The number of output features (date offset and start time).
            verbose (int): Verbosity mode for training. 0=silent, 1=progress bar, 2=one line per epoch.
            model_path (str or None): The directory path to save/load model weights.
            model (tf.keras.Model or None): The neural network model instance.
        """
        
        # Hyperparameters
        self.batch_size: int = 32
        self.num_epochs: int = 100
        self.learning_rate: float = 0.001
        self.input_size: int = 0
        self.output_size: int = 2  # Predicting date offset and start time
        self.verbose: int = 0 # Verbosity mode -> 0: silent, 1: progress bar, 2: one line per epoch
        self.model_path = None
        self.model = None
        
    def getModel(self):
        """
        Creates and compiles a neural network model.
        This function initializes a neural network model with the specified input and output sizes,
        compiles it using the Adam optimizer with a custom learning rate, and applies a custom loss function.
        It also loads the model weights if they are available.
        
        Returns:
            tf.keras.Model: The compiled neural network model.
        """
        
        model = NeuralNet(self.input_size, self.output_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=custom_loss(self.penalty_factor))
        self.load_weights()
        return model
    
    def __callback(self):
        """
        Callback function for managing model checkpoints and cleanup.
            This function handles the creation of model checkpoint directories, 
            the removal of old checkpoints, and the saving of the best model 
            based on validation loss during training.
            
            Features:
                - Creates the model path directory if it doesn't exist.
                - Removes model checkpoints older than 30 days.
                - Saves the best model based on validation loss.
            
            Usage:
                This function is intended to be used as a callback in the training 
                process of a neural network model using TensorFlow/Keras.
            
            Returns:
                tf.keras.callbacks.ModelCheckpoint: A Keras callback for saving the best model based on validation loss.
        """
        
        # Create the model path if it doesn't exist
        if not self.model_path: return None
        os.makedirs(self.model_path, exist_ok=True)
        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        
        # Remove checkpoints older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        for filename in os.listdir(self.model_path):
            file_path = os.path.join(self.model_path, filename)
            if os.path.isfile(file_path):
                file_date_str = filename.split('.')[0]
                try:
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d-%H-%M-%S-%f")
                    if file_date < cutoff_date:
                        os.remove(file_path)
                except ValueError:
                    continue
        
        # Save the best model based on validation loss
        return tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, f'{date}.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0
        )
        
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the neural network model using the provided training and validation data.
        This method initializes the model, sets the input size based on the training data,
        and trains the model using the specified batch size and number of epochs. It also
        includes a callback for additional functionality during training.
        
        Parameters:
            X_train (numpy.ndarray): The training data features.
            y_train (numpy.ndarray): The training data labels.
            X_val (numpy.ndarray): The validation data features.
            y_val (numpy.ndarray): The validation data labels.
            
        Returns:
            model (tf.keras.Model): The trained neural network model.
            history (tf.keras.callbacks.History): The history object containing details about the training process.
        """
        
        self.input_size = X_train.shape[1]
        self.model = self.getModel()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=self.verbose,
            validation_data=(X_val, y_val),
            callbacks=[self.__callback()]
        )
        
        return self.model, history
    
    def adjust_prediction(self, predictions: np.ndarray, order_created_at_list: list[datetime], task_duration_list: list[float|int], product_operation_id_list: list[int], scaler=None) -> list[dict]:
        """
        Adjusts predictions to ensure they fall within business hours and avoid overlaps.
        This function processes a list of predictions and adjusts their start and end times to ensure they:
            - Start no earlier than 09:00 and end no later than 17:00.
            - Do not overlap with each other.
            - Do not start on weekends.
            - Split tasks that span multiple days or go beyond business hours.
            
        Parameters:
            predictions (np.ndarray): An array of predicted date offsets and start times.
            order_created_at_list (list[datetime]): A list of order creation dates corresponding to each prediction.
            task_duration_list (list[float|int]): A list of task durations in seconds.
            product_operation_id_list (list[int]): A list of product operation IDs corresponding to each prediction.
            scaler (optional): A scaler object used to inverse transform the predictions.
            
        Returns:
            list[dict]: A list of dictionaries, each containing adjusted task details:
                - 'product_operation_id': The ID of the product operation.
                - 'task_duration': The duration of the task in seconds.
                - 'start_at': The adjusted start time of the task.
                - 'end_at': The adjusted end time of the task.
                - 'order_created_at': The original order creation date.
        """
        
        adjusted_pred: list[dict] = []
        last_task_end_at = None  # Keep track of the end time of the previous task

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

            # Check for overlap with the previous task
            if last_task_end_at and start_at < last_task_end_at:
                start_at = last_task_end_at  # Adjust start time to avoid overlap

            # Calculate end time
            end_at = start_at + timedelta(seconds=int(task_duration))

            # Split tasks if they go beyond business hours or span multiple days
            while start_at.date() != end_at.date() or end_at.time() > datetime.strptime("17:00", "%H:%M").time():
                # Calculate the end of the current business day
                end_of_day = start_at.replace(hour=17, minute=0, second=0, microsecond=0)

                # If the task spans multiple days or goes beyond business hours, split it
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

            # Ensure the task doesn't start on a weekend
            while start_at.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                start_at = start_at + timedelta(days=(7 - start_at.weekday()))  # Move to next Monday
                start_at = start_at.replace(hour=9, minute=0, second=0, microsecond=0)
                end_at = start_at + timedelta(seconds=int(task_duration))

            # Create final task card
            task_card = {
                'product_operation_id': product_operation_id,
                'task_duration': task_duration,
                'start_at': start_at,
                'end_at': end_at,
                'order_created_at': order_created_at_list[idx]
            }
            adjusted_pred.append(task_card)

            # Update last_task_end_at to avoid overlap for the next task
            last_task_end_at = end_at

        # After assigning times to all tasks, ensure no overlaps exist by re-sorting and adjusting
        for idx in range(1, len(adjusted_pred)):
            prev_task = adjusted_pred[idx - 1]
            curr_task = adjusted_pred[idx]

            if curr_task['start_at'] < prev_task['end_at']:
                curr_task['start_at'] = prev_task['end_at']  # Reschedule to start after the previous task's end
                curr_task['end_at'] = curr_task['start_at'] + timedelta(seconds=int(curr_task['task_duration']))

                # Ensure tasks still fall within business hours after rescheduling
                if curr_task['end_at'].time() > datetime.strptime("17:00", "%H:%M").time():
                    end_of_day = curr_task['start_at'].replace(hour=17, minute=0, second=0, microsecond=0)
                    task_duration_today = (end_of_day - curr_task['start_at']).total_seconds()

                    # Adjust for the current day, split task for the next day
                    curr_task['end_at'] = end_of_day

                    # Move the remaining duration to the next day
                    next_start_at = curr_task['start_at'] + timedelta(days=1)
                    next_start_at = next_start_at.replace(hour=9, minute=0, second=0, microsecond=0)
                    next_task_card = {
                        'product_operation_id': curr_task['product_operation_id'],
                        'task_duration': curr_task['task_duration'] - task_duration_today,
                        'start_at': next_start_at,
                        'end_at': next_start_at + timedelta(seconds=int(curr_task['task_duration'] - task_duration_today)),
                        'order_created_at': curr_task['order_created_at']
                    }
                    adjusted_pred.append(next_task_card)

        # Sort adjusted predictions by start time, in case rescheduled tasks were added later
        adjusted_pred.sort(key=lambda x: x['start_at'])
        
        return adjusted_pred
        
    def predict(self, X: np.ndarray, order_created_at_list: list[datetime], task_duration_list: list[float|int], product_operation_id_list:list[int], scaler=None) -> list[dict]:
        """
        Predict method for generating predictions using the trained model.
            This method takes in input features and additional contextual information to generate predictions
            using a pre-trained neural network model. It also adjusts the predictions based on the provided
            contextual information to ensure they are realistic and meaningful.
            
            Parameters:
                X (np.ndarray): The input features for prediction, expected to be a 2D array.
                order_created_at_list (list[datetime]): A list of datetime objects representing the creation times of orders.
                task_duration_list (list[float|int]): A list of task durations corresponding to each input sample.
                product_operation_id_list (list[int]): A list of product operation IDs corresponding to each input sample.
                scaler (optional): An optional scaler object for scaling the input features.
                
            Returns:
                list[dict]: A list of dictionaries containing the adjusted predictions for each input sample.
        """
        
        self.input_size = X.shape[1]
        self.model = self.getModel()
        predictions = self.model.predict(X, verbose=0)
        adjusted_predictions = self.adjust_prediction(predictions, order_created_at_list, task_duration_list, product_operation_id_list, scaler)
        return adjusted_predictions
    
    def load_weights(self) -> tf.keras.Model:
        """
        Loads the latest model weights from the specified checkpoint directory.
        This function checks for the latest checkpoint in the directory specified by `self.model_path`.
        If a checkpoint is found, it loads the weights into the model instance stored in `self.model`.
        This is useful for resuming training from the last saved state or for using a pre-trained model
        for inference.
        
        Returns:
            tf.keras.Model: The model instance with the loaded weights.
        """
        
        latest_checkpoint = tf.train.latest_checkpoint(self.model_path)
        if latest_checkpoint:
            self.model.load_weights(latest_checkpoint)
        return self.model
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> list[float]:
        """
        Evaluates the performance of the neural network model on the provided dataset.
        This method checks if the model is initialized. If not, it initializes the model using the input size
        derived from the provided dataset. It then evaluates the model's performance on the given features (X)
        and target labels (y).
        
        Parameters:
            X (pd.DataFrame): The input features for evaluation.
            y (pd.DataFrame): The target labels for evaluation.
            
        Returns:
            list[float]: A list of evaluation metrics, typically including loss and other metrics defined during model compilation.
        """
        
        if not self.model:
            self.input_size = X.shape[1]
            self.model = self.getModel()
        return self.model.evaluate(X, y, verbose=0)
