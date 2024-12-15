from pymongo import MongoClient
from typing import List
from  api_extract_prep.mongodb_operations.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

class MongoDBHandler:
    """
    A class to handle MongoDB operations, including connecting to a collection,
    inserting data, and reading data.

    Attributes:
        uri (str): The URI for connecting to MongoDB, sourced from the configuration file.
        db_name (str): The name of the database in MongoDB, sourced from the configuration file.
        collection_name (str): The name of the collection in MongoDB, sourced from the configuration file.
        collection (Collection): The MongoDB collection object used for performing operations.
    """

    def __init__(self):
        """
        Initialize the MongoDBHandler using details from the configuration file.
        
        This method establishes a connection to the MongoDB collection
        based on the configuration settings for the URI, database, and collection name.
        """
        self.uri = MONGO_URI
        self.db_name = MONGO_DB
        self.collection_name = MONGO_COLLECTION
        self.collection = self.connect_to_mongo()

    def connect_to_mongo(self):
        """
        Establish a connection to the MongoDB collection.

        This method connects to the MongoDB instance using the connection string provided
        in the configuration file, and returns the collection object for performing operations.

        Returns:
            Collection: MongoDB collection object for performing operations.
        """
        client = MongoClient(self.uri)
        db = client[self.db_name]
        collection = db[self.collection_name]
        return collection

    def insert_data(self, data: List[dict]) -> None:
        """
        Inserts multiple documents into the MongoDB collection.

        Args:
            data (list): A list of dictionaries representing documents to insert. Each dictionary should
                         contain key-value pairs corresponding to the fields in the MongoDB collection.

        Example:
            data = [{'propertyCode': '105376770', 'price': 249000.0, 'rooms': 2}, ...]
            handler.insert_data(data)

        If the data is empty, the method will print "No data to insert."
        """
        if data:
            result = self.collection.insert_many(data)
            print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
        else:
            print("No data to insert.")

    def read_data(self) -> List[dict]:
        """
        Reads all documents from the MongoDB collection.

        Returns:
            list: A list of documents from the collection. Each document is represented as a dictionary.

        Example:
            documents = handler.read_data()
            print(documents)  # Outputs the list of documents

        This method retrieves all documents from the MongoDB collection and returns them as a list.
        """
        documents = list(self.collection.find())
        print(f"Retrieved {len(documents)} documents from MongoDB.")
        return documents
