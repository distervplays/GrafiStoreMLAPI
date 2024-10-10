from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from datetime import datetime
from bson import ObjectId
import pandas as pd
import numpy as np

class Database:
    def __init__(self, username: str, password: str) -> None:
        self._user, self._password = quote_plus(username), quote_plus(password)
        self._uri = f"mongodb+srv://{self._user}:{self._password}@grafistoreapi.c0wyu.mongodb.net/"
        self.client = MongoClient(self._uri, server_api=ServerApi('1'))
        self.client.admin.command('ping')
        self.db = self.client['grafistore']

    def get_collection(self, collection_name):
        return self.db[collection_name]
    
    def to_json(self) -> list[dict]:
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