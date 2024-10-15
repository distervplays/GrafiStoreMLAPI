from . import Database
from pymongo import InsertOne
from bson import ObjectId

class TaskCards(Database):
    def __init__(self, username: str, password: str) -> None:
        """
        Initializes the TaskCards class with user credentials and sets up the task cards collection.
            This constructor method initializes the TaskCards class by inheriting from a parent class
            that requires username and password for authentication. It also sets up a MongoDB collection
            named 'task_cards' for further operations.
            
            Parameters:
                username (str): The username for authentication.
                password (str): The password for authentication.
            Attributes:
                collection (Collection): The MongoDB collection for task cards.
        """
        
        super().__init__(username, password)
        self.collection = self.get_collection('task_cards')
        
    def insert(self, task_card: dict) -> ObjectId:
        """
        Inserts a task card document into the collection.
        This method takes a dictionary representing a task card and inserts it into the MongoDB collection.
        It returns the ObjectId of the inserted document, which can be used to reference the document in the database.
        
        Parameters:
            task_card (dict): A dictionary containing the task card data to be inserted into the collection.
            
        Returns:
            ObjectId: The unique identifier of the inserted task card document.
        """
        
        return self.collection.insert_one(task_card).inserted_id
        
    def insert_many(self, task_card: list) -> list[ObjectId]:
        """
        Inserts multiple task card documents into the collection.
        This method takes a list of task card dictionaries and inserts them into the MongoDB collection.
        It returns a list of ObjectIds representing the IDs of the inserted documents.
        
        Parameters:
            task_card (list): A list of dictionaries, where each dictionary represents a task card document to be inserted.
            
        Returns:
            list[ObjectId]: A list of ObjectIds corresponding to the inserted task card documents.
        """
        
        return self.collection.insert_many(task_card).inserted_ids
        
    def find(self, filter: dict) -> dict:
        """
        Finds a single document in the collection that matches the given filter criteria.
        This method searches the MongoDB collection for a document that matches the specified filter.
        If a matching document is found, it is returned as a dictionary. If no matching document is found,
        an empty dictionary is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria for the search. The keys and values
                           in this dictionary should correspond to the fields and values to be matched in the
                           collection.
                        
        Returns:
            dict: A dictionary representing the found document if a match is found, otherwise an empty dictionary.
        """
        
        return dict(self.collection.find_one(filter)) if self.collection.find_one(filter) else {}
        
    def find_many(self, filter: dict) -> list:
        """
        Retrieves multiple documents from the collection based on the provided filter criteria.
        This method queries the database collection using the specified filter and returns a list of documents
        that match the filter criteria. If no documents match the filter, an empty list is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria for querying the collection.
            
        Returns:
            list: A list of documents that match the filter criteria. If no documents are found, an empty list is returned.
        """
        
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_all(self) -> list:
        """
        Retrieves all documents from the collection.
        This method queries the MongoDB collection associated with the instance and returns all documents
        as a list. If the collection is empty, it returns None.
        
        Returns:
            list: A list of all documents in the collection, or None if the collection is empty.
        """
        
        return list(self.collection.find({})) if self.collection.find({}) else None