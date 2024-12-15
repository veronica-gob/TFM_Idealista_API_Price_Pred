# TFM_Idealista_API_Price_Pred

## Project Overview

This repository contains the comprehensive implementation of a machine learning model for predicting property prices using data from Idealista's API. The main goal of the project is to build an Machine Learning Model that can predict property prices based on various features and identify undervalued properties, defined as those with actual listing prices at least 10% lower than the predicted prices.

## Author

**Verónica García de Olalla Blancafort**  

Master of Science in Data Science, Universitat Oberta de Catalunya (UOC)

Email: vgdob133@gmail.com  or vgarcia_de_olalla@uoc.edu

GitHub: https://github.com/veronica-gob

## Date
December 2024

## Project Description

This project focuses on predicting property prices in the real estate market using machine learning. The methodology involves extracting data from the Idealista API, cleaning and transforming it, training machine learning models for price prediction, and identifying undervalued properties. These properties are flagged for investment opportunities when their actual prices are at least 10% lower than the predicted prices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/veronica-gob/TFM_Idealista_API_Price_Pred.git
   cd TFM_Idealista_API_Price_Pred
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Idealista API credentials: 
- Go to [Idealista API](https://developers.idealista.com/access-request) and register for an API key. Once registered, you will receive an API key and API password.
   
4. Download and install MongoDB Compass:
- Download [(MongoDB Compass)](https://www.mongodb.com/try/download/compass) and install MongoDB Compass and connect to your local or remote MongoDB server using the connection string provided by MongoDB.


## Configuration

- **Idealista API Key**: You will need to provide the API credentials to authenticate and fetch property data from the Idealista API. These credentials should be added to your `config.py` file in the `api_extract_prep/idealista_api_extractor` folder.

- **MongoDB**: To configure MongoDB in this project, modify with your connection settings in `config.py` in the `api_extract_prep/mongodb_operations`.


## Running the Project

1. **Data Collection, Exploration, and Cleaning**: Data is extracted from the Idealista API, stored in MongoDB, transformed to remove inconsistencies and identify useful patterns and saved in  `api_extract_prep/preprocessed_data.csv`.

    ```bash
    python api_extract_prep/main_api_extract_prep.py
    ```

    Data can be only extracted form idealista and saved in MongoDB with:

    ```bash
    python api_extract_prep/idealista_api_extractor/extract_data.py
    ```

2. **Machine Learning Models and Hyperparameter Tuning**: Multiple machine learning have been tested for price prediction:
   - **Linear Regression**
   - **Decision Tree Regression**
   - **Random Forest Regression**
   - **Gradient Boosting Regression**
   - **XGB Regression**

   Hyperparameter tuning is performed using grid search or random search to find the best settings for each model, using R² (coefficient of determination). The hyperparameter tunning can be found on a Jupyter Notebook: `parameter_tuning_modelling_comparisson.ipynb`.

3. **Model Evaluation and Comparison and Undervalued Property Identification **: After hyperparameter tunning, the models are evaluated in train and test with Cross Validation. The results are saved in `modelling_comparisson_metrics/model_comparisson_metrics.xlsx` for easier comparison.  With the selected model, property prices are predicted and undervalued properies are filtered. Properties with actual prices at least 10% lower than the predicted prices are considered undervalued and saved in `predict_powerbi/undervalued_properties.csv`.

    ```bash
    python modelling/main_evaluate_and_predict.py
    ```

5. **Interactive Dashboard Development**: A Power BI dashboard has been created to visualize undervalued properties and facilitate decision-making for investors in `predict_powerbi/Undervalued Barcelona Properties.pbix`, using the csv file saved in the same folder.

## Project Structure

```plaintext
TFM_IDEALISTA_API_PRICE_PRED/
│
├── api_extract_prep/                # API Data extraction, loading to Mongo DB, and transformation
│   ├── idealista_api_extractor/     
│   │   ├── config.py                # Idealista API configuration settings
│   │   ├── filters.py               # Data filtering logic
│   │   ├── extract_data.py          # API extraction module
│   ├── eda_and_transformations/   
│   │   ├── eda_and_transformations.ipynb # Jupyter notebook for EDA and transformations
│   │   ├── idealista_pricem2_district.csv # Example dataset that should be downloaded Idealista Website
│   │   ├── transform_data.py        # Data transformation module
│   ├── mongodb_operations/    
│   │   ├── config.py                # MongoDB configuration settings
│   │   ├── insert_read_data.py      # MongoDB insert and read operations
│   ├── preprocessed_data/           # Folder for saving preprocessed data, after runing the following main script
│   ├── main_api_extract_prep.py     # Main script to orchestrate extraction, loading, and transformations
│   
├── modelling/                                      # Machine learning models and evaluation
│   ├── parameter_tuning_modelling_comparison.ipynb # Hyperparameter tuning notebook
│   ├── train_and_evaluate_models.ipynb             # Model evaluation and comparison
│   ├── modelling_comparison_metrics/               # Folder to save model evaluation metrics
│   ├── train_and_predict_selected_model.py         # Final model price prediction and undervalued property filtering
│   ├── predict-powerbi/                            # Folder to save undervalued properties and Power Bi Dashboard
│   │   ├── undervalued_properties.csv  
│   │   ├── Undervalued Barcelona Properties.pbix
│   ├── main_evaluate_and_predict.py                # Main script for model evaluation and predictions
│   
├── docs/                                  # Documentation files
│   │   ├── idealista_api_docs             # Idealista Doc Files
│   │   │   ├── oauth2-documentation.pdf  
│   │   │   ├── property-search-api-v3.5.pdf    
│   │   ├── _build/html                   # Sphinx Documentation
│   │   │   ├── index.html
│   │   │   ├── modules.html
│   │   │   ├── ...
```

## Documentation

The project documentation is available in HTML format, created with Sphinx. You can view it [here](docs/_build/html/index.html).
The Idealista API documentation has also been saved in the docs folder, the [OAuth2 Documentation](docs/idealista_api_docs/oauth2-documentation.pdf) and the [Property Search API Documentation](docs/idealista_api_docs/property-search-api-v3_5.pdf)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
