from . import Database
from pymongo import InsertOne
from bson import ObjectId

class Products(Database):
    def __init__(self, username: str, password: str) -> None:
        super().__init__(username, password)
        self.collection = self.get_collection('products')
        
    def insert_product(self, product: dict) -> ObjectId:
        return self.collection.insert_one(product).inserted_id
    
    def insert_products(self, products: list[dict]) -> list[ObjectId]:
        return self.collection.insert_many(products).inserted_ids
        
    def find(self, filter: dict) -> dict:
        return dict(self.collection.find_one(filter)) if self.collection.find_one(filter) else {}
        
    def find_many(self, filter: dict) -> list:
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_products(self) -> list:
        return list(self.collection.find({})) if self.collection.find({}) else None