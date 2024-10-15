from urllib.parse import urljoin
from dotenv import load_dotenv
from modules.PremoAPI import PremoAPI
from web import db, orders, products, taskcards, operations
from bson import ObjectId
from web import api
import os
load_dotenv()

def update_database(learning_days: int = 7) -> None:
    """
    Updates the database with new learning data.
    This function fetches learning data from an API and updates the database with new orders, products, task cards, and product operations. It ensures that orders are not duplicated by checking for existing order IDs in the database. The function performs bulk inserts for task cards and operations to optimize database performance.
    Features:
    - Fetches learning data from an API.
    - Checks for existing orders to avoid duplication.
    - Inserts new orders, products, task cards, and product operations into the database.
    - Performs bulk inserts for task cards and operations.
    
    Parameters:
        learning_days (int): The number of days to consider for learning data. Default is 7.
    """
    
    api.learning_days = 7
    data = api.learn_data

    taskcards_bulk = []
    operations_bulk = []

    for order in data:
        if orders.find({'order_id': order['order_id']}): continue
        order_dict = {k: v for k, v in order.items() if k in ['order_id', 'delivery_date', 'order_created_at']}
        order_id = orders.insert(order_dict)
        
        for prod in order['products']:
            products_dict = {k: v for k, v in prod.items() if k in ['material', 'color']}
            products_dict['order_id'] = order_id
            product_id = products.insert(products_dict)
            
            taskcards_arr: list = [{**taskcard, 'product_id': product_id} for taskcard in prod['task_cards']]
            taskcards_bulk.extend(taskcards_arr)
            
            operations_arr = [{**operation, 'product_id': product_id} for operation in prod['product_operations']]
            operations_bulk.extend(operations_arr)
            

    # Bulk insert taskcards and operations
    if len(taskcards_bulk) > 0 and len(operations_bulk) > 0:
        _: list[ObjectId] = taskcards.insert_many(taskcards_bulk)
        _: list[ObjectId] = operations.insert_many(operations_bulk)