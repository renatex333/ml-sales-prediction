# Project: Sales Prediction Model with Random Forest

This project focuses on building and using a machine learning model to predict sales using a Random Forest Regressor. The project consists of two main components:

1. **Training the Model:** Data is extracted from a PostgreSQL database using a user-provided SQL script, and the model is trained on this data.
2. **Predicting Sales:** The trained model is used to predict sales on new data, with predictions saved back to the database.

The project employs environment variables for secure database connection management and uses Python libraries such as `pandas`, `SQLAlchemy`, and `scikit-learn`.

## Features

- **Data Ingestion:** Extracts data from a PostgreSQL database using an SQL script.
- **Random Forest Model Training:** Trains a Random Forest Regressor on historical data.
- **Sales Prediction:** Uses the trained model to predict future sales.
- **Model and Prediction Management:** The trained model is saved in serialized form, and predictions are written back to the database.

## How It Works

### 1. Training the Model
The training process involves:
- **Database Connection:** Using `dotenv` to load environment variables for securely connecting to the PostgreSQL database.
- **Data Loading:** The SQL script specified by the user is executed to retrieve the data.
- **Model Training:** A Random Forest Regressor is trained on the extracted dataset.
- **Model Saving:** The trained model is serialized and saved as a `.pkl` file in the `models` directory.

### 2. Predicting Sales
The prediction process involves:
- **Model Loading:** The previously trained model is loaded from the saved `.pkl` file.
- **Data Loading:** The new data for which predictions are to be made is fetched using a provided SQL script.
- **Prediction and Storage:** Predictions are generated using the model and stored back into the PostgreSQL database in a specified table.

## Usage

### To Train the Model

```bash
python3 train.py <path_to_sql_script>
```

- `<path_to_sql_script>`: Path to the SQL file that retrieves the training data from the PostgreSQL database.

Example:

```bash
python3 train.py data/train.sql
```

The trained model will be saved in the `models` directory.

### To Predict Sales

```bash
python3 predict.py <path_to_model> <path_to_sql_script>
```

- `<path_to_model>`: Path to the saved model file (`.pkl`).
- `<path_to_sql_script>`: Path to the SQL script that retrieves the data for prediction.

Example:

```bash
python3 predict.py models/model.pkl data/predict.sql
```

Predictions will be saved back to the PostgreSQL database in the table specified by the environment variables.

## Environment Variables

The project uses environment variables for database credentials and configuration, stored in a `.env` file. Ensure the following variables are defined:

- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host
- `DB_PORT`: Database port
- `DB_NAME`: Database name
- `DB_PREDICTION_TABLE`: Table where predictions will be stored
- `DB_PREDICTION_SCHEMA`: Schema where the predictions table will be stored

## Dependencies

The project relies on the following Python libraries:

- `pandas`
- `SQLAlchemy`
- `scikit-learn`
- `psycopg2`
- `python-dotenv`

To install them, run:

```bash
pip install -r requirements.txt
```
