from flask import Blueprint, jsonify, request
from Helpers.DataPreProcessing import preprocess_data
from utils.FlattenData import flatten_input
from utils.UpdateDatabase import update_database
from sklearn.model_selection import train_test_split
from . import api, agent, db, orders, products, operations, taskcards
from pymongo import MongoClient
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

@views.route('/train', methods=['POST', 'GET'])
def train():
    """
    Handles the training and evaluation of a machine learning model.
    This function manages the training process of a machine learning model, including data preprocessing,
    splitting the data into training and testing sets, and handling both POST and GET requests for training
    and evaluation respectively.
    
    - For POST requests:
        - Sets the training parameters (number of epochs, batch size, learning rate, verbose) from the request JSON.
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
        agent.verbose = request.json.get('verbose', 0)
        
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
        
@views.route('/database', methods=['GET', 'POST', 'DELETE', 'PUT'])
def database_handler():
    """Handles database operations for different HTTP request methods.
    
    GET:
        Retrieves all records from the specified table.
        - Query Parameters:
            - table (str): The name of the table to retrieve records from.
                           valid values: 'orders', 'products', 'operations', 'taskcards'
        - Returns:
            - JSON response containing the records with ObjectIds converted to strings and renamed fields.
            
    POST:
        Inserts a new record into the specified table.
        - JSON Body:
            - to_table (str): The name of the table to insert the record into.
            - Other fields as required by the table schema.
        - Returns:
            - JSON response indicating success and the index of the inserted record.
            
    DELETE:
        Deletes a record from the specified table.
        - JSON Body:
            - from_table (str): The name of the table to delete the record from.
            - idx (str): The index of the record to delete.
        - Returns:
            - JSON response indicating success and the index of the deleted record.
            
    PUT:
        Updates a record in the specified table.
        - JSON Body:
            - from_table (str): The name of the table to update the record in.
            - idx (str): The index of the record to update.
            - update (dict): The fields to update with their new values.
        - Returns:
            - JSON response indicating success and the index of the updated record.
            
    Returns:
        - JSON response indicating an invalid request method if the method is not GET, POST, DELETE, or PUT.
    """
    
    if request.method == 'GET':
        fromTable = request.args.get('table', None)
        if not fromTable: return {'message': 'No table specified!', 'status': 400}
        
        table_classes = {
            'orders': orders,
            'products': products,
            'operations': operations,
            'taskcards': taskcards
        }
        
        if not fromTable in table_classes:
            return {'message': 'Invalid table specified!', 'status': 400}
        table = table_classes[fromTable]
        
        # Convert ObjectIds to strings for JSON serialization
        table_dict: list[dict] = table.get_all()
        for record in table_dict:
            record['_id'] = str(record['_id'])
            
            # Convert order_id and product_id to strings for JSON serialization
            if fromTable == 'products':
                record['order_id'] = str(record['order_id'])
            if fromTable in ['operations', 'taskcards']:
                record['product_id'] = str(record['product_id'])
        
        # Rename the _id fields to idx for consistency
        table_dict = [{'idx': record.pop('_id'), **record} for record in table_dict]
        
        if fromTable == 'products':
            table_dict = [{'order_idx': record.pop('order_id'), **record} for record in table_dict] 
        if fromTable in ['operations', 'taskcards']:
            table_dict = [{'product_idx': record.pop('product_id'), **record} for record in table_dict] 
        
        return jsonify(table_dict)
    
    if request.method == 'POST':
        data = request.json
        to_table = data.get('to_table', None)
        
        if not to_table: 
            return {'message': 'No table specified!', 'status': 400}
        
        table_classes = {
            'orders': orders,
            'products': products,
            'operations': operations,
            'taskcards': taskcards
        }
        
        if not to_table in table_classes:
            return {'message': 'Invalid table specified!', 'status': 400}
        table = table_classes[to_table]
        idx = table.insert(data)
        return {'message': f'Data added successfully! (Index: {idx})'}
    
    if request.method == 'DELETE':
        data = request.json
        from_table = data.get('from_table', None)
        idx = data.get('idx', None)
        
        if not from_table: return {'message': 'No table specified!', 'status': 400}
        if not idx: return {'message': 'No index specified!', 'status': 400}
        
        table_classes = {
            'orders': orders.collection,
            'products': products.collection,
            'operations': operations.collection,
            'taskcards': taskcards.collection
        }
        
        if not from_table in table_classes:
            return {'message': 'Invalid table specified!', 'status': 400}
        collection: MongoClient = table_classes[from_table]
        collection.delete_one({'_id': idx})
        
        return {'message': f'Data deleted successfully! (Index: {idx})'}
    
    if request.method == 'PUT':
        data = request.json
        from_table = data.get('from_table', None)
        idx = data.get('idx', None)
        update = data.get('update', None)
        
        if not from_table: return {'message': 'No table specified!', 'status': 400}
        if not idx: return {'message': 'No index specified!', 'status': 400}
        if not update: return {'message': 'No update specified!', 'status': 400}
        
        table_classes = {
            'orders': orders.collection,
            'products': products.collection,
            'operations': operations.collection,
            'taskcards': taskcards.collection
        }
        
        if not from_table in table_classes:
            return {'message': 'Invalid table specified!', 'status': 400}
        collection: MongoClient = table_classes[from_table]
        collection.update_one({'_id': idx}, {'$set': update})
        
        return {'message': f'Data updated successfully! (Index: {idx})'}
    
    return {'message': 'Invalid request method!', 'status': 400}
    
