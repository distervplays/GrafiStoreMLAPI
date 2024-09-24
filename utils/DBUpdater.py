from urllib.parse import quote_plus
from modules.database_tables.Orders import Orders
from modules.database_tables.Products import Products
from modules.database_tables.TaskCards import TaskCards
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import json
import os

class DBUpdater:
    def __init__(self) -> None:
        load_dotenv()
        username = os.getenv("mongodb_user")
        password = os.getenv("mongodb_password")
        self._orders = Orders(username, password)
        self._products = Products(username, password)
        self._taskcards = TaskCards(username, password)
        self._history_distance = 7
        
    def update(self) -> None:
        """Method to update the database with the latest data from the API.
        
        :returns: None
        :rtype: None
        """
        data_url = f"https://premo.gshub.nl/api/dinand/orders/{self._history_distance}"
        data = json.loads(requests.get(data_url).text)
        
        # Pre-fetch all existing orders and products IDs once
        existing_orders = {doc['order_id']: doc['_id'] for doc in self._orders.get_orders()}
        existing_products = {doc['order_id']: doc['_id'] for doc in self._products.get_products()}
        
        for order in data:
            if len(order['products']['task_cards']) == 0:
                continue

            # Normalize order and drop unnecessary product fields
            orderDF = pd.json_normalize(order)
            orderDF.drop(columns=[col for col in orderDF.columns if col.startswith("products.")], inplace=True)
            orderDF.rename(columns={'id': 'order_id'}, inplace=True)
            orderDF['delivery_date'] = pd.to_datetime(orderDF['delivery_date'], errors='coerce')
            orderDF['order_created_at'] = pd.to_datetime(orderDF['order_created_at'], errors='coerce')
            
            # Convert DataFrame to dictionary format
            order_dict = orderDF.to_dict(orient='records')[0]
            order_id = existing_orders.get(order_dict['order_id'])

            if order_id is None:
                # If order doesn't exist, insert it and store the inserted ID
                order_id = self._orders.insert_order(order_dict)
                existing_orders[order_dict['order_id']] = order_id
            
            # Prepare product details (excluding task_cards) and check if it already exists
            products_dict = {'order_id': order_id, **{k: v for k, v in order['products'].items() if k != 'task_cards'}}
            product_id = existing_products.get(order_id)

            if product_id is None:
                # If product doesn't exist, insert it and store the inserted ID
                product_id = self._products.insert_product(products_dict)
                existing_products[order_id] = product_id

            # Normalize task cards, insert product_id and convert timestamps
            tasksDF = pd.json_normalize(order['products'], 'task_cards')
            tasksDF.insert(0, 'product_id', product_id)
            tasksDF['task_time_start'] = pd.to_datetime(tasksDF['task_time_start'], errors='coerce')
            tasksDF['task_time_end'] = pd.to_datetime(tasksDF['task_time_end'], errors='coerce')
            
            tasks_dict = tasksDF.to_dict(orient='records')
            
            # Insert task cards if not already present
            if not self._taskcards.find({'product_id': product_id}):
                self._taskcards.insert_taskcards(tasks_dict)         
                
        
    
        
        
        
        
        