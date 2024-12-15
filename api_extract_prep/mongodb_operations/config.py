"""
Configuration file for MongoDB.

This file contains the necessary configuration settings to connect to
MongoDB for storing data fetched from the Idealista API.

Attributes:
    MONGO_URI (str): The URI for connecting to MongoDB (e.g., 'mongodb://localhost:27017').
    MONGO_DB (str): The name of the MongoDB database (e.g., 'idealista_api_db').
    MONGO_COLLECTION (str): The MongoDB collection name where data will be stored (e.g., 'bcn_homes_2411').
"""

MONGO_URI='mongodb://localhost:27017'
MONGO_DB='idealista_api_db'
MONGO_COLLECTION='bcn_homes_2412' # change name each month