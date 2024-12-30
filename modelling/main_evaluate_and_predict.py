import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from api_extract_prep.mongodb_operations.insert_read_data import MongoDBHandler
from train_and_evaluate_models import ModelTrainer
from train_and_predict_selected_model import GradientBoostingTrainPredict
from datetime import datetime

def main():
    """
    Main function to train multiple regression models, evaluate their performance, and 
    predict property prices and filtering undervalued properties using Gradient Boosting. The function performs the following steps:
    
    Steps:
        1. **Load and Prepare Data**: The data is loaded from a CSV file, and the feature and target variables are defined.
        2. **Define Models**: The function sets up parameters for multiple models (Decision Tree, Random Forest, Gradient Boosting, XGBoost) for training.
        3. **Train and Evaluate Models**: Models are trained, evaluated, and their metrics are saved to an Excel file using the `ModelTrainer` class.
        4. **Predict Property Prices**: A new dataset is loaded (with the new monthly 5,000 listings extracted from Idealista API), and the best-performing model (Gradient Boosting) is used to predict property prices.
        5. **Predict Undervalued Properties**: The undervalued properties are filtered (where actual price is 10% lower than the predicted price) and saved to a CSV file.

    Arguments:
        None

    Returns:
        None

    Example:
        Running the `main()` function will:
            - Load and preprocess training data,
            - Train multiple regression models,
            - Evaluate model performance and save metrics to a file,
            - Predict prices and filter undervalued properties using the Gradient Boosting model and save results to a CSV file.

    Example usage:
        main()

    This function requires:
        - `train_and_evaluate_models.ModelTrainer`: To train and evaluate models.
        - `train_and_predict_selected_model.GradientBoostingTrainPredict`: To predict prices and filter undervalued properties.
        - `api_extract_prep.mongodb_operations.MongoDBHandler`: To extract new monthly dataset.
    """
    
    
    # Define the target column and feature columns
    target_column = "price"
    features = ['numPhotos', 'size', 'distance', 'bathrooms', 'hasLift', 'price_m2_oct24', 'amenity.score', 'floor', 
                'propertyType', 'highlight.groupDescription', 'exterior', 'parkingSpace.hasParkingSpace']

    # Read training preprocessed data
    df = pd.read_csv('api_extract_prep/preprocessed_data/prep_data_2024-12-06.csv') 
    
    # Prepare features (X) and target variable (y)
    X = df[features]
    y = df[target_column]

    # Define models parameters
    models_params = { 
        'decision_tree': {
            'model': DecisionTreeRegressor(
                min_samples_split=30,  
                min_samples_leaf=15,  
                max_depth=6           
            ), 
            'name': 'Decision Tree'
        },
        'random_forest': {
            'model': RandomForestRegressor(
                n_estimators=100,     
                min_samples_split=10, 
                min_samples_leaf=5,   
                max_depth=15,         
                max_features='log2'   
            ), 
            'name': 'Random Forest'
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(
                subsample=0.7,        
                n_estimators=150,      
                min_samples_split=10,  
                min_samples_leaf=5, 
                max_depth=5,         
                learning_rate=0.05   
            ), 
            'name': 'Gradient Boosting'
        },
        'xgboost': {
            'model': XGBRegressor(
                subsample=0.8,        
                reg_lambda=1,          
                reg_alpha=0.1,         
                n_estimators=150,     
                max_depth=4,           
                learning_rate=0.1,    
                colsample_bytree=0.6   
            ), 
            'name': 'XGBoost'
        }
    }
    
    # Define metrics file path directly
    metrics_file_path = 'modelling/modelling_comparisson_metrics/model_comparison_metrics.xlsx'
    
    # Initialize ModelTrainer and train models
    model_trainer = ModelTrainer(metrics_file_path)
    model_trainer.run_linear_regression(X, y)  # Run linear regression model
    model_trainer.run_other_models(X, y, models_params)  # Run other models

    print(f"Evaluation metrics saved to {metrics_file_path}")

    # Initialize MongoDB handler for fetching data
    mongo_handler = MongoDBHandler()  

    # Read new data (df_new) that you want to predict on
    current_date = datetime.now().strftime("%Y-%m-%d") 
    df_new = pd.read_csv(f'api_extract_prep/preprocessed_data/prep_data_{current_date}.csv') 
    
    # Define the output file path where results should be saved
    output_filename = 'modelling/predict_powerbi/undervalued_properties.csv'

    # Create an instance of the GradientBoostingTrain_Predict class
    model_predictor = GradientBoostingTrainPredict(target_column, features, mongo_handler, output_filename)
    
    # Call the method to train, predict, filter undervalued properties, and save the result to the CSV
    model_predictor.save_undervalued_prop(df, df_new)

if __name__ == "__main__":
    main()
