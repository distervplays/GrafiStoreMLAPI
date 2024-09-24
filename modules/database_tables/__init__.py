from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from datetime import datetime

class Database:
    def __init__(self, username: str, password: str) -> None:
        self._user, self._password = quote_plus(username), quote_plus(password)
        self._uri = f"mongodb+srv://{self._user}:{self._password}@grafistoreapi.c0wyu.mongodb.net/?retryWrites=true&w=majority&appName=GrafistoreAPI"
        
        self.client = MongoClient(self._uri, server_api=ServerApi('1'))
        self.client.admin.command('ping')
            
        self.db = self.client['grafistore']

    def get_collection(self, collection_name):
        return self.db[collection_name]