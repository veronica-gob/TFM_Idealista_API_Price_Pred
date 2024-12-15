"""
Filter file for the Idealista API.

This file contains the necessary filters for API extract from the Idealista API.

Attributes:
    - base_url (str): The base URL of the Idealista API.
    - country (str): The country code for the search (e.g., 'es' for Spain).
    - operation (str): The operation type (e.g., 'sale' for sales listings).
    - max_items (str): Maximum number of items to fetch per request (items per page, maximum 50).
    - order (str): The order in which results should be sorted.
    - center (str): Latitude and longitude of the search center point.
    - distance (str): Search radius in meters.
    - property_type (str): Type of property (e.g., 'homes', 'offices', 'premises').
    - sort (str): Sorting criteria (e.g., 'asc' or 'desc').
    - maxprice (str): Maximum price filter.
    - language (str): Language for the results (e.g., 'es' for Idealista.com).

Note:
    Refer to the documentation folder for further information.
"""

base_url = 'https://api.idealista.com/3.5/'     
country = 'es'         
language = 'es'    
max_items = '50'    
operation = 'sale'     
property_type = 'homes'    
order = 'priceDown'    
center = '41.3851,2.1734'    
distance = '10000'     
sort = 'asc'    
# bankOffer = 'false'
maxprice = '800000' 