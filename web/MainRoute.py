from flask import Blueprint, jsonify, request
from Helpers.DataPreProcessing import preprocess_data
from utils.FlattenData import flatten_input
from utils.UpdateDatabase import update_database
from sklearn.model_selection import train_test_split
from . import api, agent
import threading

views = Blueprint(
    name='main',
    import_name=__name__,
    url_prefix='/'
)

@views.route('/', methods=['POST'])
def home():
    """
    Handles the home route for the API.
    This function processes HTTP requests to the home route. It supports both GET and POST methods.
    - For GET requests, it returns a welcome message.
    - For POST requests, it expects a JSON payload containing the number of learning days, updates the database accordingly, 
      and returns a success message. If an error occurs during this process, it returns an error message.
    
    Usage:
        - Send a GET request to receive a welcome message.
        - Send a POST request with a JSON payload containing 'learning_days' to update the database.
        
    Returns:
        dict: A dictionary containing a message indicating the result of the request.
    """
    
    if request.method == 'POST':
        try:
            learning_days = request.json['learning_days']
            update_database(int(learning_days))
            return {'message': 'Database updated successfully!'}
        except:
            return {'message': 'An error occurred!'}
    return {'message': 'Welcome to the Premo API!'}

@views.route('/play_data')
def flat_data():
    """
    Flattens the play data and returns it as a JSON response.
    This function retrieves play data from the API, flattens it using the `flatten_input` function, 
    and then returns the flattened data as a JSON response using Flask's `jsonify`.
    
    Usage:
        This function is typically used in a Flask route to process and return play data in a flattened format.
        
    Returns:
        Response: A Flask `jsonify` response containing the flattened play data.
    """
    
    flat_play: list[dict] = flatten_input(api.play_data)
    return jsonify(flat_play)

@views.route('/train', methods=['POST', 'GET'])
def train():
    """
    Handles the training and evaluation of a machine learning model.
    This function manages the training process of a machine learning model, including data preprocessing,
    splitting the data into training and testing sets, and handling both POST and GET requests for training
    and evaluation respectively.
    
    - For POST requests:
        - Sets the training parameters (number of epochs, batch size, learning rate, penalty factor) from the request JSON.
        - Starts the training process in a separate thread to avoid blocking.
        
    - For GET requests:
        - Retrieves the model and evaluates it on the test data.
        - Returns the loss if the model has been trained, otherwise returns a message indicating the model is not trained yet.
        
    Usage:
            - Send a POST request to start training the model with specified parameters.
            - Send a GET request to evaluate the model and get the loss on the test data.
            
    Returns:
            dict: A dictionary containing a message indicating the training status or the evaluation loss.
    """
    
    _, X, y, _ = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if request.method == 'POST':
        agent.num_epochs = request.json.get('num_epochs', 100)
        agent.batch_size = request.json.get('batch_size', 16)
        agent.learning_rate = request.json.get('learning_rate', 0.001)
        agent.penalty_factor = request.json.get('penalty_factor', 0.5)
        
        def train_model():
            agent.train(X_train, y_train, X_test, y_test)

        training_thread = threading.Thread(target=train_model)
        training_thread.start()
        
        return {'message': 'Training started!'}
    
    if request.method == 'GET':
        agent.getModel()
        try:
            loss = agent.evaluate(X_test, y_test)
            return {'loss': loss}
        except:
            return {'message': 'Model not trained yet!'}

    
    