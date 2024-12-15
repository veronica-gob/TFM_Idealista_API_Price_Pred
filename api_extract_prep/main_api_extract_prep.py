import pandas as pd
from api_extract_prep.eda_and_transformations.transform_data import TransformData
from api_extract_prep.idealista_api_extractor.extract_data import IdealistaAPI
from datetime import datetime
import requests
import base64
import json
import time
from api_extract_prep.idealista_api_extractor.filters import *
from api_extract_prep.idealista_api_extractor.config import *
from api_extract_prep.mongodb_operations.insert_read_data import MongoDBHandler


from mongodb_operations.insert_read_data import MongoDBHandler  # Assuming this reads data from MongoDB

def main():
    """
    Main function to orchestrate the data extraction, transformation, and saving process.

    This function performs the following steps:
    1. Extracts data from the Idealista API (currently commented out).
    2. Reads data from MongoDB using MongoDBHandler.
    3. Creates a DataFrame from the MongoDB data.
    4. Applies transformations to the data using the TransformData class.
    5. Saves the transformed data to a CSV file with the current date in the filename.

    The transformed data is saved in the 'api_extract_prep/preprocessed_data/' directory.

    Steps:
        1. **Extract Data**: The `IdealistaAPI` class is used to extract data from the Idealista API and save it to MongoDB. This step is currently commented out.
        2. **Read Data**: The `MongoDBHandler` is used to read the extracted data from MongoDB.
        3. **Transform Data**: The data is processed using the `TransformData` class to apply necessary transformations (e.g., handling duplicates).
        4. **Save Transformed Data**: The transformed data is saved to a CSV file, named with the current date for easy identification.

    Example:
        Running the `main()` function will:
            - Fetch the data from MongoDB,
            - Transform it according to the specified rules,
            - Save it to a CSV file in the 'preprocessed_data' directory.
    """
    #Step 1: Extract data from the API and save it to MongoDB
    # api_handler = IdealistaAPI()  # Initialize Idealista API handler
    # api_handler.extract_api_data()  # Extract data and save to MongoDB

    # Step 2: Read data from MongoDB
    mongo_handler = MongoDBHandler()  # Initialize MongoDB handler
    mongo_data = mongo_handler.read_data()  # Fetch all documents from MongoDB collection
    
    # Step 3: Create a DataFrame from the MongoDB data
    df = pd.json_normalize(mongo_data)  # Normalize nested JSON to flat table

    # Step 4: Apply transformations using the TransformData class
    transformer = TransformData(df)  # Instantiate the transformer with the DataFrame
    # Apply all transformations, in this step we are removing duplicated and other transformations
    # In case duplicate properties are added in previous steps
    preprocessed_df = transformer.transform()  

    # Step 5: Save the transformed data to a CSV file with the current date in the file name
    current_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date in YYYY-MM-DD format
    dir = f'api_extract_prep/preprocessed_data/prep_data_{current_date}.csv'
    preprocessed_df.to_csv(dir, index=False)
    print(f"Data transformation complete and saved to '{dir}'.")

if __name__ == "__main__":
    main()  