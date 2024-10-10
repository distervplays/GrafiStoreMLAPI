from urllib.parse import urljoin
from dotenv import load_dotenv
from modules.PremoAPI import PremoAPI
from modules.database import Database
from modules.database.Orders import Orders
from modules.database.Products import Products
from modules.database.TaskCards import TaskCards
from modules.database.ProductOperations import Operations
from bson import ObjectId
import pandas as pd
import numpy as np
import requests
import json
import os
load_dotenv()


api = PremoAPI()
api.learning_days = 7
data = api.learn_data

username = os.getenv("mongodb_user")
password = os.getenv("mongodb_password")

db = Database(username, password)
orders = Orders(username, password)
products = Products(username, password)
taskcards = TaskCards(username, password)
operations = Operations(username, password)

orders_bulk = []
products_bulk = []
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
    tk_ids: list[ObjectId] = taskcards.insert_many(taskcards_bulk)
    op_ids: list[ObjectId] = operations.insert_many(operations_bulk)