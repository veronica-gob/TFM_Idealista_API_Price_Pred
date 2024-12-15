import requests
import base64
import json
import time
from  api_extract_prep.idealista_api_extractor.filters import *
from api_extract_prep.idealista_api_extractor.config import *
import os 
import sys
from api_extract_prep.mongodb_operations.insert_read_data import MongoDBHandler
from pymongo import MongoClient
from typing import List
from  api_extract_prep.mongodb_operations.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

class IdealistaAPI:
    """
    A class to handle interactions with the Idealista API, including authentication,
    data retrieval, and storing data in MongoDB.

    Attributes:
        api_key (str): The API key for authenticating with Idealista API.
        secret (str): The secret key for authenticating with Idealista API.
        mongo_handler (MongoDBHandler): An instance of the MongoDBHandler class used for
                                        interacting with MongoDB.
    """
    def __init__(self):
        """
        Initializes the IdealistaAPI class with the provided API credentials and MongoDB handler.
        """
        self.api_key = API_KEY
        self.secret = SECRET
        self.mongo_handler = MongoDBHandler()

    def get_oauth_token(self) -> str:
        """
        Retrieves an OAuth token from the Idealista API using the provided API key and secret.

        This method constructs a base64 encoded authorization header and sends a POST request
        to the Idealista API to obtain an access token.

        Returns:
            str: The access token for API authentication.

        Raises:
            Exception: If the API request fails or the response does not contain an access token.
        """
        message = self.api_key + ":" + self.secret   
        auth = "Basic " + base64.b64encode(message.encode("ascii")).decode("ascii")
        headers = {
            "Authorization": auth,
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }
        params = {"grant_type": "client_credentials", "scope": "read"}

        response = requests.post("https://api.idealista.com/oauth/token", headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve OAuth token: {response.status_code} - {response.text}")

        return json.loads(response.text)['access_token']
    
    def create_search_url(self) -> str:
        """
        Constructs a search URL by combining the base URL and search parameters for Idealista API.

        This function takes predefined variables (such as operation type, country, maximum items, etc.)
        and combines them into a properly formatted URL string, which can be used to send a search
        request to the Idealista API.

        Returns:
            str: The formatted search URL for sending requests to the Idealista API.
        """
        url = (base_url +      
            country +
            '/search?operation=' + operation +
            '&maxItems=' + max_items +
            '&order=' + order +
            '&center=' + center +
            '&distance=' + distance +
            '&propertyType=' + property_type +
            '&sort=' + sort + 
            '&numPage=%s' +
            '&maxPrice=' + maxprice +
            '&language=' + language)
        
        return url

    def fetch_data(self, url: str) -> dict:
        """
        Sends a request to the Idealista API and retrieves data for a specific page.

        Args:
            url (str): The search URL with pagination to fetch property data.

        Returns:
            dict: The JSON response from the API, containing property data.

        Raises:
            Exception: If the API request fails or the response is not as expected.
        """
        token = self.get_oauth_token()
        headers = {
        'Content-Type': 'Content-Type: multipart/form-data;',   
        'Authorization': 'Bearer ' + token
    }
        response = requests.post(url, headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return {}

        return json.loads(response.text)

    def extract_api_data(self) -> None:
        """
        Extracts property data from the Idealista API and stores it in MongoDB.

        This method fetches data page by page, retrieves property information, and inserts it
        into MongoDB.

        If the data extraction fails or there is no data, the process will terminate gracefully.

        Raises:
            Exception: If there is an error during data extraction or database insertion.
        """
        url = self.create_search_url()
        page = 1
        initial_results = self.fetch_data(url % page)
        if 'totalPages' not in initial_results:
            print("No data found or an error occurred.")
            return

        total_pages = min(initial_results['totalPages'], page + 95)

        for current_page in range(page, total_pages + 1):
            paginated_url = url % current_page
            results = self.fetch_data(paginated_url)

            if 'elementList' in results:
                properties = results['elementList']
                print(f"Fetched {len(properties)} properties from page {current_page}.")

                try:
                    self.mongo_handler.insert_data(properties)
                    print(f"Inserted {len(properties)} documents from page {current_page} into MongoDB.")
                except Exception as e:
                    print(f"Failed to insert data from page {current_page}: {e}")
            else:
                print(f"No property data found on page {current_page}.")
            time.sleep(1)  # To respect API rate limits


if __name__ == "__main__":
    """
    Main entry point for the script that runs the IdealistaAPI class to extract and store property data.
    """
    api_handler = IdealistaAPI()
    api_handler.extract_api_data()
