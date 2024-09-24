from . import Database
from pymongo import InsertOne
from bson import ObjectId


class Orders(Database):
    def __init__(self, username: str, password: str) -> None:
        super().__init__(username, password)
        self.collection = self.get_collection('orders')
        
    def insert_order(self, order: dict) -> ObjectId:
        return self.collection.insert_one(order).inserted_id
    
    def insert_orders(self, orders: list[dict]) -> list[ObjectId]:
        return self.collection.insert_many(orders).inserted_ids
    
    def find(self, filter: dict) -> dict:
        return dict(self.collection.find_one(filter)) if self.collection.find_one(filter) else {}
        
    def find_many(self, filter: dict) -> list:
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_orders(self) -> list:
        return list(self.collection.find({})) if self.collection.find({}) else []
    
    