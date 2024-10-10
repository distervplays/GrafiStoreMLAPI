from . import Database
from pymongo import InsertOne
from bson import ObjectId

class TaskCards(Database):
    def __init__(self, username: str, password: str) -> None:
        super().__init__(username, password)
        self.collection = self.get_collection('task_cards')
        
    def insert(self, task_card: dict) -> ObjectId:
        return self.collection.insert_one(task_card).inserted_id
        
    def insert_many(self, task_card: list) -> list[ObjectId]:
        return self.collection.insert_many(task_card).inserted_ids
        
    def find(self, filter: dict) -> dict:
        return dict(self.collection.find_one(filter)) if self.collection.find_one(filter) else {}
        
    def find_many(self, filter: dict) -> list:
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_all(self) -> list:
        return list(self.collection.find({})) if self.collection.find({}) else None