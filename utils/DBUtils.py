from urllib.parse import quote_plus
from modules.database_tables.Orders import Orders
from modules.database_tables.Products import Products
from modules.database_tables.TaskCards import TaskCards
from modules.database_tables.ProductOperations import Operations
from dotenv import load_dotenv
from bson import ObjectId
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
        self._operations = Operations(username, password)
        self._history_distance = 7
        
    def update(self) -> None:
        """Method to update the database with the latest data from the API.
        
        :returns: None
        :rtype: None
        """
        
        learn_url = f"https://premo.gshub.nl/api/dinand/learn/{self._history_distance}"
        data = requests.get(learn_url).json()
        
        taskcards_bulk = []
        operations_bulk = []

        for order in data:
            order_dict = {k: v for k, v in order.items() if k in ['order_id', 'delivery_date', 'order_created_at']}
            if self._orders.find({'order_id': order_dict['order_id']}): continue
            order_id = self._orders.insert(order_dict)
            
            for prod in order['products']:
                products_dict = {k: v for k, v in prod.items() if k in ['material', 'color']}
                products_dict['order_id'] = order_id
                product_id = self._products.insert(products_dict)
                
                taskcards_arr = [{**taskcard, 'product_id': product_id} for taskcard in prod['task_cards']]
                taskcards_bulk.extend(taskcards_arr)
                
                operations_arr = [{**operation, 'product_id': product_id} for operation in prod['product_operations']]
                operations_bulk.extend(operations_arr)
                

        # Bulk insert taskcards and operations
        if len(taskcards_bulk) > 0 and len(operations_bulk) > 0:
            tk_ids: list[ObjectId] = self._taskcards.insert_many(taskcards_bulk)
            op_ids: list[ObjectId] = self._operations.insert_many(operations_bulk)       
            
def json_from_database(orders: Orders, products: Products, taskcards: TaskCards, operations: Operations) -> list[dict]:
    # Convert lists to DataFrames

    orders_df = pd.json_normalize(orders.get_all()) # pd.DataFrame(orders.get_all())
    products_df = pd.DataFrame(products.get_all()).rename(columns={'_id': 'product_id'})
    taskcards_df = pd.DataFrame(taskcards.get_all()).drop(columns=['_id'])
    operations_df = pd.DataFrame(operations.get_all()).drop(columns=['_id'])

    # Merge DataFrames
    merged_df = orders_df.merge(products_df, left_on='_id', right_on='order_id', suffixes=('_order', '_product'))
    merged_df = merged_df.merge(taskcards_df, left_on='product_id', right_on='product_id', suffixes=('', '.taskcard'))
    merged_df = merged_df.merge(operations_df, left_on='product_id', right_on='product_id', suffixes=('.taskcard', '.operation'))
    merged_df = merged_df.rename(columns={'start_at': 'start_at.taskcard', 'end_at': 'end_at.taskcard'})
    merged_df = merged_df.rename(columns={'order_id_order': 'order_id'})
    merged_df = merged_df.drop(columns=['order_id_product'])
    merged_df.head()


    # Group by order and product
    grouped = merged_df.groupby(['_id', 'delivery_date', 'order_created_at', 'material', 'color'])
    # Create the final data structure
    data = []
    for (order_id, delivery_date, order_created_at, material, color), group in grouped:
        group = group.drop(columns=['_id', 'product_id', 'order_created_at', 'material', 'color'])
        
        product_operations = group[['task_title.operation', 'task_duration.operation', 'sort_order.operation', 'workspace.operation']].drop_duplicates()
        product_operations = product_operations.rename(columns=lambda x: x.replace('.operation', ''))
        product_operations = product_operations.to_dict('records')
        
        task_cards = group[[v for v in group.columns if '.taskcard' in v]].drop_duplicates()
        task_cards = task_cards.rename(columns=lambda x: x.replace('.taskcard', ''))
        task_cards = task_cards.to_dict('records')
        
        data.append({
            "order_id": int(group['order_id'].iloc[0]),  # Use order_id from orders_df
            "delivery_date": delivery_date,
            "order_created_at": order_created_at,
            "products": [{
                "material": material,
                "color": color,
                "product_operations": product_operations,
                "task_cards": task_cards
            }]
        })
    return data
