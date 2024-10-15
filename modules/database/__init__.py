from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from datetime import datetime
from bson import ObjectId
import pandas as pd
import numpy as np

class Database:
    def __init__(self, username: str, password: str) -> None:
        """
        Database connection initialization class.
            This class handles the initialization of a connection to a MongoDB database using the provided
            username and password. It sets up the connection URI, creates a MongoClient instance, and pings
            the database to ensure the connection is established. The database client is then stored for
            further operations.
            
            - Initializes the database connection with the provided credentials.
            - Constructs the MongoDB URI with the encoded username and password.
            - Creates a MongoClient instance with the specified server API version.
            - Pings the database to verify the connection.
            - Stores the database client for further operations.
            
            Parameters:
                username (str): The username for the MongoDB database.
                password (str): The password for the MongoDB database.
                
            Attributes:
                _user (str): The URL-encoded username.
                _password (str): The URL-encoded password.
                _uri (str): The constructed MongoDB URI.
                client (MongoClient): The MongoClient instance for database operations.
                db (Database): The database instance for the specified database.
                
        """
        
        self._user, self._password = quote_plus(username), quote_plus(password)
        self._uri = f"mongodb+srv://{self._user}:{self._password}@grafistoreapi.c0wyu.mongodb.net/"
        self.client = MongoClient(self._uri, server_api=ServerApi('1'))
        self.client.admin.command('ping')
        self.db = self.client['grafistore']

    def get_collection(self, collection_name: str) -> MongoClient:
        """
        Fetches a specific collection from the database.
        This function is used to retrieve a collection from the database instance associated with the class.
        It allows for dynamic access to different collections within the database by specifying the collection name.
        
        Parameters:
            collection_name (str): The name of the collection to retrieve from the database.
            
        Returns:
            Collection: The collection object corresponding to the specified collection name.
        """
        
        return self.db[collection_name]
    
    def to_json(self) -> list[dict]:
        """
        Converts database collections to a JSON-serializable list of dictionaries.
            This function fetches data from multiple MongoDB collections ('orders', 'products', 'task_cards', 'operations')
            and organizes it into a nested JSON structure. It minimizes database calls by fetching all data at once and 
            uses lookup dictionaries to efficiently map related data.
            
            - Fetches all documents from the 'orders', 'products', 'task_cards', and 'operations' collections.
            - Creates lookup dictionaries for products, tasks, and operations to map them to their respective orders.
            - Constructs a JSON-serializable list of dictionaries where each order contains its related products, 
              and each product contains its related operations and task cards.
              
            Returns:
                list[dict]: A list of dictionaries where each dictionary represents an order with its related products, 
                            operations, and task cards.
        """
        
        json_data: list[dict] = []
        ord_collection = self.get_collection('orders')
        prod_collection = self.get_collection('products')
        tasks_collection = self.get_collection('task_cards')
        op_collection = self.get_collection('operations')

        # Fetch all data at once to minimize database calls
        orders = list(ord_collection.find({}))
        products = list(prod_collection.find({}))
        tasks = list(tasks_collection.find({}))
        operations = list(op_collection.find({}))

        # Create lookup dictionaries for products, tasks, and operations
        products_lookup = {}
        for product in products:
            order_id = product['order_id']
            if order_id not in products_lookup:
                products_lookup[order_id] = []
            products_lookup[order_id].append(product)

        tasks_lookup = {}
        for task in tasks:
            product_id = task['product_id']
            if product_id not in tasks_lookup:
                tasks_lookup[product_id] = []
            tasks_lookup[product_id].append(task)

        operations_lookup = {}
        for operation in operations:
            product_id = operation['product_id']
            if product_id not in operations_lookup:
                operations_lookup[product_id] = []
            operations_lookup[product_id].append(operation)

        for order in orders:
            data = {k: v for k, v in order.items() if k != '_id'}
            order_id = order['_id']
            
            data['products'] = []
            if order_id in products_lookup:
                for product in products_lookup[order_id]:
                    prod_data = {'product_id': str(product['_id'])}
                    prod_data.update({k: v for k, v in product.items() if k not in ['_id', 'order_id']})
                    prod_data['product_operations'] = []
                    prod_data['task_cards'] = []
                    
                    if product['_id'] in operations_lookup:
                        prod_data['product_operations'] = [{k:v for k, v in op.items() if k not in ['_id', 'product_id']} for op in operations_lookup[product['_id']]]
                    if product['_id'] in tasks_lookup:
                        prod_data['task_cards'] = [{k:v for k, v in task.items() if k not in ['_id', 'product_id']} for task in tasks_lookup[product['_id']]]

                    data['products'].append(prod_data)
            
            json_data.append(data)
        
        return json_data