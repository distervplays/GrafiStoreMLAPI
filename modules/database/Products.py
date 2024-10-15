from . import Database
from pymongo import InsertOne
from bson import ObjectId

class Products(Database):
    def __init__(self, username: str, password: str) -> None:
        """
        Initializes the Products class with user credentials and sets up the products collection.
            This constructor method initializes the Products class by taking a username and password,
            passing them to the superclass initializer, and setting up the 'products' collection
            from the database.
            
            Parameters:
                username (str): The username for database authentication.
                password (str): The password for database authentication.
            Attributes:
                collection (Collection): The MongoDB collection for 'products'.
        """
        
        super().__init__(username, password)
        self.collection = self.get_collection('products')
        
    def insert(self, product: dict) -> ObjectId:
        """
        Inserts a new product document into the collection.
        This method takes a dictionary representing a product and inserts it into the MongoDB collection.
        It returns the ObjectId of the inserted document, which can be used to reference the product in the database.
        
        Parameters:
            product (dict): A dictionary containing the product details to be inserted into the collection.
            
        Returns:
            ObjectId: The unique identifier of the inserted product document.
        """
        
        return self.collection.insert_one(product).inserted_id
    
    def insert_many(self, products: list[dict]) -> list[ObjectId]:
        """
        Inserts multiple product documents into the collection.
        This method takes a list of product dictionaries and inserts them into the MongoDB collection.
        Each dictionary in the list represents a product document to be inserted. The method returns
        a list of ObjectIds corresponding to the inserted documents.
        
        Parameters:
            products (list[dict]): A list of dictionaries, where each dictionary represents a product document
                                   to be inserted into the collection.
                                
        Returns:
            list[ObjectId]: A list of ObjectIds corresponding to the inserted product documents.
        """
        
        return self.collection.insert_many(products).inserted_ids
        
    def find(self, filter: dict) -> dict:
        """
        Method to find a single document in the collection based on a filter.
        This method searches the MongoDB collection for a single document that matches the specified filter.
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
        Method to find multiple documents in the collection based on a filter.
        This method queries the database collection for documents that match the specified filter criteria.
        If documents matching the filter are found, they are returned as a list. If no documents match the
        filter, an empty list is returned.
        
        Parameters:
            filter (dict): A dictionary specifying the filter criteria for querying the collection.
            
        Returns:
            list: A list of documents that match the filter criteria. If no documents are found, an empty list is returned.
            
        Usage:
            products = Products()
            filter_criteria = {"category": "electronics"}
            matching_products = products.find_many(filter_criteria)
        """
        
        return list(self.collection.find(filter)) if self.collection.find(filter) else []
        
    def get_all(self) -> list:
        """
        Retrieves all documents from the collection.
        This method queries the MongoDB collection associated with the instance and retrieves all documents.
        It returns a list of documents if any are found, otherwise it returns None.
        
        Returns:
            list: A list of all documents in the collection if any exist, otherwise None.
        """
        return list(self.collection.find({})) if self.collection.find({}) else None