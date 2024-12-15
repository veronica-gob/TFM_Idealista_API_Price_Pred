import pandas as pd
from api_extract_prep.mongodb_operations.insert_read_data import MongoDBHandler  # Import your MongoDB handler here
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostingTrainPredict:
    """
    This class uses a Gradient Boosting Regressor to predict property prices based on input features. 
    It trains the model using historical data, predicts the prices for new data, filters undervalued 
    properties (where actual price is 10% lower than the predicted price), enriches the undervalued properties dataset with
    additional data from MongoDB (adding the non-processed columns previously transformed), and saves the final result to a CSV file for further analysis.
    """
    def __init__(self, target_column: str, features: list, mongo_handler: MongoDBHandler, output_filename: str):
        """
        Initializes the GradientBoostingTrainPredict class.
        
        Parameters
        ----------
        target_column : str
            The name of the target column for prediction.
        
        features : list
            List of feature column names for training and prediction.
        
        mongo_handler : MongoDBHandler
            MongoDB handler instance for fetching MongoDB data.
        
        output_filename : str
            Path to save the final dataframe to CSV.
        """
        self.target_column = target_column
        self.features = features
        self.mongo_handler = mongo_handler
        self.output_filename = output_filename
        self.model = GradientBoostingRegressor(
            subsample=0.7,
            n_estimators=150,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=5,
            learning_rate=0.05
        )
    
    def train_and_predict(self, df: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Train the GradientBoostingRegressor model on the provided dataset and generate predictions
        for the new dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training dataset containing features and target.
        
        df_new : pd.DataFrame
            New dataset to predict on.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with additional column 'price_prediction'.
        """
        # Ensure no missing values in the feature set
        df = df.dropna(subset=self.features + [self.target_column])
        df_new = df_new.dropna(subset=self.features)
        
        # Separate features and target
        X_train = df[self.features]
        y_train = df[self.target_column]
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Generate predictions for df_new
        X_new = df_new[self.features]
        df_new['price_prediction'] = self.model.predict(X_new)
        
        # Retain the original price column if it exists
        if self.target_column not in df_new.columns:
            df_new[self.target_column] = None  # Fill in with NaN if the column doesn't exist
        
        return df_new

    def save_undervalued_prop(self, df: pd.DataFrame, df_new: pd.DataFrame) -> None:
        """
        Process the data, filter undervalued properties, and join with MongoDB data, then save to CSV.
        
        Parameters
        ----------
        df : pd.DataFrame
            The training data.
        
        df_new : pd.DataFrame
            The new data for prediction.
        
        Returns
        -------
        None
            This method does not return any value. It saves the final dataframe to a CSV file.
        """
        # Train and predict using the model
        df_new = self.train_and_predict(df, df_new)
        
        # Filter undervalued properties where actual price is more than 10% lower than predicted
        df_undervalued_prop = df_new[df_new['price'] < (df_new['price_prediction'] * 0.90)]
        
        # Fetch MongoDB data
        mongo_data = self.mongo_handler.read_data()  # Fetch all documents from MongoDB collection
        
        # Create a DataFrame from the MongoDB data
        df_new_no_transformed = pd.json_normalize(mongo_data)  # Normalize nested JSON to flat table
        df_new_no_transformed = df_new_no_transformed.drop_duplicates(subset=['propertyCode', 'price'])
        
        # Add prefix 'transf_' to all columns except 'property_code' and 'price'
        df_undervalued_prop = df_undervalued_prop.rename(columns=lambda col: 'transf_' + col if col not in ['propertyCode', 'price', 'price_prediction'] else col)
        
        # Perform a left join with df_new_no_transformed on 'property_code'
        # Convert 'propertyCode' to integer before the merge
        df_undervalued_prop['propertyCode'] = df_undervalued_prop['propertyCode'].astype(int)
        df_new_no_transformed['propertyCode'] = df_new_no_transformed['propertyCode'].astype(int)

        # Perform the left join on 'propertyCode' and 'price'
        df_joined = df_undervalued_prop.merge(df_new_no_transformed, on=['propertyCode', 'price'], how='left')
                
        # Save the final joined DataFrame to a CSV file
        df_joined.to_csv(self.output_filename, index=False)
        print(f"File saved to {self.output_filename}")