from . import Database
from pymongo import InsertOne
from bson import ObjectId
from .Products import Products


class Orders(Database):
    def __init__(self, username: str, password: str) -> None:
        """
        Initialize the Orders class with user credentials and set up the orders collection.
        This constructor method initializes the Orders class by inheriting from a parent class
        that requires a username and password for authentication. It also sets up the 'orders'
        collection from the database for further operations.
        
        Parameters:
            username (str): The username for database authentication.
            password (str): The password for database authentication.
            
        Attributes:
            collection (Collection): The MongoDB collection for 'orders'.
            
        """
        
        super().__init__(username, password)
        self.collection = self.get_collection('orders')
        
    def insert(self, order: dict) -> ObjectId:
        """
        Inserts a new order into the database collection.
        This method takes a dictionary representing an order and inserts it into the MongoDB collection.
        It returns the ObjectId of the inserted document, which can be used to reference the order in the database.
        
        Parameters:
            order (dict): A dictionary containing the order details to be inserted into the database.
            
        Returns:
            ObjectId: The unique identifier of the inserted order document.
        """
        
        return self.collection.insert_one(order).inserted_id
    
    def insert_many(self, orders: list[dict]) -> list[ObjectId]:
        """
        Inserts multiple order documents into the database collection.
        This method takes a list of order dictionaries and inserts them into the MongoDB collection
        associated with this instance. It returns a list of ObjectIds representing the IDs of the 
        inserted documents.
        
        Parameters:
            orders (list[dict]): A list of dictionaries, where each dictionary represents an order 
                                 document to be inserted into the collection.
                                
        Returns:
            list[ObjectId]: A list of ObjectIds corresponding to the inserted order documents.
        """
        
        return self.collection.insert_many(orders).inserted_ids
    
    def find(self, filter: dict) -> dict:
        """
        Finds a single document in the collection that matches the given filter.
        This method searches the database collection for a document that matches the specified filter criteria.
        If a matching document is found, it is returned as a dictionary. If no matching document is found, 
        an empty dictionary is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria for the search. The keys and values 
                           in the filter dictionary should correspond to the fields and values to be matched 
                           in the collection.
                        
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
        This method queries the MongoDB collection associated with the instance and retrieves all documents.
        If the collection is empty, it returns an empty list.
        
        Returns:
            list: A list of all documents in the collection. If the collection is empty, an empty list is returned.
            
        Usage:
            This function is used to fetch all records from the MongoDB collection. It can be useful for
            retrieving data for display, analysis, or further processing.
            
        Example:
            orders = Orders()
            all_orders = orders.get_all()
            print(all_orders)
        """
        
        return list(self.collection.find({})) if self.collection.find({}) else []
    
    