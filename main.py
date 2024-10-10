from utils.UpdateDatabase import *
from modules.database import Database    
from urllib.parse import urljoin
from dotenv import load_dotenv
import requests
import json
import os
load_dotenv()

# db = Database(os.getenv("mongodb_user"), os.getenv("mongodb_password"))

# json_data = db.to_json()
# print(json.dumps(json_data, indent=4))
# print(len(json_data))
# with open('data/dataset.json', 'w') as f:
#     json.dump(json_data, f, indent=4)

# with open('data/dataset.json', 'r') as f:
#     data = json.load(f)

# for order in data:
#     products = order['products']
#     if len(products) == 2:
#         with open('data.json', 'w') as f:
#             json.dump(order, f, indent=4)
#         break