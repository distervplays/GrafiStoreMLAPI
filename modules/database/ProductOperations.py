from . import Database
from pymongo import InsertOne
from bson import ObjectId

class Operations(Database):
    def __init__(self, username: str, password: str) -> None:
        """
        Initializes the ProductOperations class with user credentials and sets up the database collection.
            This constructor method initializes the ProductOperations class by inheriting from a parent class
            that requires username and password for authentication. It also sets up a specific collection
            within the database for product operations.
            
            Parameters:
                username (str): The username for database authentication.
                password (str): The password for database authentication.
            Attributes:
                collection (Collection): The MongoDB collection for product operations.
        """
        
        super().__init__(username, password)
        self.collection = self.get_collection('operations')
        
    def insert(self, operation: dict) -> ObjectId:
        """
        Inserts a new document into the collection.
        This method takes a dictionary representing a document and inserts it into the MongoDB collection.
        It returns the ObjectId of the inserted document, which can be used to reference the document later.
        
        Parameters:
            operation (dict): A dictionary representing the document to be inserted into the collection.
            
        Returns:
            ObjectId: The unique identifier of the inserted document.
        """
        
        return self.collection.insert_one(operation).inserted_id
    
    def insert_many(self, operations: list[dict]) -> list[ObjectId]:
        """
        Inserts multiple documents into the collection.
        This method takes a list of dictionaries, each representing a document to be inserted into the MongoDB collection.
        It uses the `insert_many` method of the collection to insert all the documents at once and returns the list of 
        ObjectIds of the inserted documents.
        
        Parameters:
            operations (list[dict]): A list of dictionaries, where each dictionary represents a document to be inserted.
            
        Returns:
            list[ObjectId]: A list of ObjectIds corresponding to the inserted documents.
        """
        
        return self.collection.insert_many(operations).inserted_ids
        
    def find(self, filter: dict) -> dict:
        """
        Finds a single document in the collection that matches the given filter.
        This method searches the MongoDB collection for a document that matches the specified filter criteria.
        If a matching document is found, it is returned as a dictionary. If no matching document is found,
        an empty dictionary is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria to search for in the collection.
            
        Returns:
            dict: A dictionary representing the found document, or an empty dictionary if no document matches the filter.
        """
        
        return dict(self.collection.find_one(filter)) if self.collection.find_one(filter) else {}
        
    def find_many(self, filter: dict) -> list[dict]:
        """
        Retrieves multiple documents from the collection based on the provided filter criteria.
        This method queries the database collection using the specified filter and returns a list of 
        documents that match the filter criteria. If no documents match the filter, an empty list is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria for querying the collection.
            
        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents a document from the collection 
                        that matches the filter criteria. If no documents match, an empty list is returned.
        """
        
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_all(self) -> list[dict]:
        """
        Retrieves all documents from the collection.
        This method queries the MongoDB collection associated with the instance and returns all documents
        found in the collection as a list of dictionaries. If the collection is empty, it returns None.
        
        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a document in the collection.
                        Returns None if the collection is empty.
                        
        Usage:
            product_operations = ProductOperations()
            all_products = product_operations.get_all()
            if all_products:
                for product in all_products:
                    print(product)
            else:
                print("No products found in the collection.")
        """
        
        return list(self.collection.find({})) if self.collection.find({}) else None