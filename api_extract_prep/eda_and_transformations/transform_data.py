import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TransformData:
    """
    A class to perform various transformations on a DataFrame containing real estate data.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing raw data.

    Methods
    -------
    filters()
        Remove duplicates based on propertyCode and filter rows for specific conditions.
    drop_unnecessary_columns()
        Drop unnecessary columns from the DataFrame.
    remove_outliers()
        Remove outliers in specified columns based on the IQR method.
    join_with_district_prices(file)
        Join the DataFrame with district-level price data from a specified file.
    map_status()
        Map the 'status' column to integer values and fill missing values.
    fill_floor()
        Transform and fill the 'floor' column based on specific rules.
    map_prop_type()
        Map the 'propertyType' column to integer values and fill missing values.
    convert_bool_columns()
        Convert specified columns to boolean and then to integers.
    map_group_description()
        Map the 'highlight.groupDescription' column to numerical values.
    create_new_features()
        Create new features from existing columns.
    create_amenity_score()
        Calculate an amenity score based on the sum of boolean amenity columns.
    apply_one_hot_encoding()
        Apply one-hot encoding to categorical columns.
    transform()
        Apply all transformations in sequence.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TransformData class with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing raw data.
        """
        self.df = df

    def filters(self) -> pd.DataFrame:
        """
        Remove duplicates based on propertyCode and filter rows for specific conditions.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame after filtering.
        """
        self.df['price'] = self.df['parkingSpace.parkingSpacePrice'].fillna(0).astype(int) + self.df['price'].astype(int)
        self.df = self.df.drop_duplicates(subset='propertyCode')
        self.df = self.df[self.df['municipality'] == 'Barcelona']
        self.df = self.df[self.df['bathrooms'] != 0]
        return self.df
    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        """
        Drop unnecessary columns from the DataFrame.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame after dropping columns.
        """
        columns_to_drop = [
            "_id", "thumbnail", "externalReference", "operation", 
            "address", "country", "latitude", "longitude", 
            "url", "description", "priceInfo.price.amount", "priceInfo.price.currencySuffix", 
            "suggestedTexts.subtitle", "suggestedTexts.title", "priceInfo.price.priceDropInfo.formerPrice",
            'priceInfo.price.priceDropInfo.priceDropValue',  'topNewDevelopment', 'municipality', 'province',  'newDevelopmentFinished', 'detailedType.typology', 'detailedType.subTypology', 'priceInfo.price.priceDropInfo.priceDropPercentage',
               'parkingSpace.isParkingSpaceIncludedInPrice', 'parkingSpace.parkingSpacePrice', 'showAddress', 'priceByArea'
        ]
        self.df = self.df.drop(columns=columns_to_drop, errors='ignore')
        return self.df
    
    def remove_outliers(self) -> pd.DataFrame:
        """
        Remove outliers in specified columns based on the IQR method.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame after removing outliers.
        """
        columns_to_check = ['price', 'size']
        
        for column in columns_to_check:
            if column in self.df.columns:
                # Calculate IQR
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Filter the DataFrame
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            else:
                print(f"Warning: Column '{column}' not found in the DataFrame. Skipping outlier removal for this column.")
        
        return self.df
    
    def join_with_district_prices(self, file: str) -> pd.DataFrame:
        """
        Join the DataFrame with district-level price data from the file saved in the same folder.

        Parameters
        ----------
        file : str
            The path to the file containing district price data.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame after joining.
        """
        try:
            # Read the Excel file
            district_prices = pd.read_csv(file)
            
            # Join with the main DataFrame on the 'district' column
            self.df = self.df.merge(district_prices, on='district', how='left')
        except Exception as e:
            print(f"Error while reading or joining district price data: {e}")
        
        return self.df
    
    
    def map_status(self) -> pd.DataFrame:
        """
        Map the 'status' column to integer values and fill missing values.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with mapped 'status'.
        """
        status_mapping = {
            'good': 1,
            'renew': 2,
            'newdevelopment': 3,
        }
        self.df['status'] = self.df['status'].map(status_mapping).fillna(0).astype(int)
        return self.df
    
    def fill_floor(self) -> pd.DataFrame:
        """
        Maps letter values for floor levels and adjusts numeric floor levels according to specified rules.
        
        The function handles the following:
        - Maps letter values ('en', 'bj', 'st', 'ss') to numeric values.
        - Maps floor values greater than or equal to 7 to 7.
        - Fills missing floor values based on property type:
          - For 'chalet' and 'countryHouse', sets floor value to 8.
          - For missing floors in other property types, sets the floor value to 2.
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the adjusted 'floor' column.
        """
        floor_mapping = {
        'en': 0,  # Entresuelo
        'bj': -1,    # Bajo
        'st': -2, # Semi-sótano
        'ss': -2,   # Sótano
    }
    
        self.df['floor'] = self.df['floor'].map(floor_mapping).fillna(self.df['floor'])
        
        self.df['floor'] = pd.to_numeric(self.df['floor'], errors='coerce')

        self.df['floor'] = self.df['floor'].apply(
                    lambda x: 7 if x >= 7 else x)

        self.df['floor'] = self.df.apply(
            lambda row: 8 if row['propertyType'] in ['chalet', 'countryHouse'] 
            else 2 if pd.isnull(row['floor']) 
            else row['floor'], axis=1
        )
    
        return self.df
    
    def map_prop_type(self) -> pd.DataFrame:
        """
        Maps property type to integer values and fills missing values with 0.
        
        Property types are mapped as follows:
        - 'studio' -> 0
        - 'flat' -> 1
        - 'penthouse', 'duplex' -> 2
        - 'chalet', 'countryHouse' -> 3
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the 'propertyType' column mapped to integers.
        """
        prop_mapping = {
            'studio': 0,
            'flat': 1,
            'penthouse': 2,
            'duplex': 2,
            'chalet': 3,
            'countryHouse': 3
        }
        self.df['propertyType'] = self.df['propertyType'].map(prop_mapping).fillna(0).astype(int)
        return self.df

    
    def convert_bool_columns(self) -> pd.DataFrame:
        """
        Converts specified columns to boolean values, filling nulls with False.
        
        The columns are:
        - 'hasLift', 'hasVideo', 'has3DTour', 'has360', 'hasStaging', 'hasPlan', 
          'newDevelopment', 'topPlus', 'exterior', 'parkingSpace.hasParkingSpace'.
        
        The function converts the values in these columns to boolean values, and 
        then to integers (0 for False, 1 for True).

        Returns:
        -------
        pd.DataFrame
            The DataFrame with the specified columns converted to boolean (then integer) values.
        """
        columns_to_convert = [
            'hasLift', 'hasVideo', 'has3DTour', 'has360', 'hasStaging', 'has3DTour',
         'hasPlan', 'newDevelopment', 'topPlus', 'exterior', 'parkingSpace.hasParkingSpace'
        ]
        
        for col in columns_to_convert:
            self.df[col] = self.df[col].fillna(False).astype(bool)
        
        # Convert them to 0 and 1
        self.df = self.df.astype({col: 'int' for col in self.df.select_dtypes('bool').columns})
        return self.df
    
    def map_group_description(self) -> pd.DataFrame:
        """
        Maps group descriptions to numerical values for encoding.
        
        The mapping is as follows:
        - 'Destacado' -> 1
        - 'Top' -> 2
        - 'Top+' -> 3
        Any missing or unrecognized group description is mapped to 0.
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the 'highlight.groupDescription' column mapped to numeric values.
        """
        desc_mapping = {
            'Destacado': 1,
            'Top': 2,
            'Top+': 3,
        }
        self.df['highlight.groupDescription'] = self.df['highlight.groupDescription'].map(desc_mapping).fillna(0).astype(float)
        return self.df
    
    def create_new_features(self) -> pd.DataFrame:
        """
        Creates new features based on existing columns.
        
        The following new features are created:
        - 'rooms+bathrooms' = sum of rooms and bathrooms.
        - 'room.to.size.ratio' = ratio of rooms to size.
        - 'room+bathrooms.to.size.ratio' = ratio of rooms and bathrooms to size.
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the newly created features.
        """
        self.df['rooms+bathrooms'] = self.df['rooms'] + self.df['bathrooms']
        self.df['room.to.size.ratio'] = self.df['rooms'] / self.df['size']
        self.df['room+bathrooms.to.size.ratio'] = self.df['rooms+bathrooms'] / self.df['size']
        return self.df
    
    def create_amenity_score(self) -> pd.DataFrame:
        """
        Creates an amenity score based on the sum of boolean amenity columns.
        
        The amenity score is calculated by summing up the boolean values in the following columns:
        - 'hasLift', 'hasVideo', 'has3DTour', 'has360', 'hasStaging', 'has3DTour',
          'hasPlan', 'newDevelopment', 'topPlus', 'exterior', 'parkingSpace.hasParkingSpace'.
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the 'amenity.score' column added.
        """
        amenity_columns = [
            'hasLift', 'hasVideo', 'has3DTour', 'has360', 'hasStaging', 'has3DTour',
         'hasPlan', 'newDevelopment', 'topPlus', 'exterior', 'parkingSpace.hasParkingSpace'
        ]
        for col in amenity_columns:
            self.df[col] = self.df[col].fillna(False).astype(int)
        self.df['amenity.score'] = self.df[amenity_columns].sum(axis=1)
        return self.df
    
    def apply_one_hot_encoding(self) -> pd.DataFrame:
        """
        Applies one-hot encoding to the 'neighborhood' and 'district' columns.
        
        The function performs one-hot encoding on the specified columns, generating
        binary columns for each category (excluding the first level to avoid redundancy).
        
        Returns:
        -------
        pd.DataFrame
            The DataFrame with the 'neighborhood' and 'district' columns one-hot encoded.
        """
        columns_to_encode = [
             'neighborhood', 'district'
        ]
        self.df = pd.get_dummies(self.df, columns=columns_to_encode, drop_first=True)
        self.df = self.df.astype({col: 'int' for col in self.df.select_dtypes('bool').columns})
        return self.df

    def transform(self) -> pd.DataFrame:
        """
        Applies all the transformation functions in sequence to clean and prepare the DataFrame.
        
        The transformations include filtering, dropping columns, removing outliers, 
        mapping status and property types, filling missing values, creating new features,
        applying one-hot encoding, and more.
        
        Returns:
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        self.filters()
        self.drop_unnecessary_columns()
        self.remove_outliers()
        self.join_with_district_prices('api_extract_prep/eda_and_transformations/idealista_pricem2_district.csv')
        self.map_status()
        self.fill_floor()
        self.map_prop_type()
        self.convert_bool_columns()
        self.map_group_description()
        self.create_new_features()
        self.create_amenity_score()
        self.apply_one_hot_encoding()
        return self.df
