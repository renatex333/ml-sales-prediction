import os
import pandas as pd
from dotenv import load_dotenv
from src.loader import load_data, load_model
from sqlalchemy import create_engine

def predict(data_path: str, model_path: str) -> None:
    """
    Main function to predict the total sales.

    Args:
        data_path (str): Path to the data. Can be a path to:
            - SQL script;
            - Other options to be implemented.
        model_path (str): Path to the model with the trained model. Can be a path to:
            - Pickle file.
            - Other options to be implemented.
    """
    load_dotenv()
    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    data = load_data(data_path, "sql", db_url)
    model = load_model(model_path)
    make_predictions(data, model, engine)

def make_predictions(data: pd.DataFrame, model: object, engine: object) -> None:
    """
    Predict the total sales.

    Args:
        data (pd.DataFrame): Data to be used for prediction.
        model: Model to be used for prediction.
        engine (Engine): Database engine.
    """
    predictions = model.predict(data)
    data["prediction_total_sales"] = predictions

    save_predictions(data=data, save_mode="sql", engine=engine, table=os.getenv("DB_PREDICTION_TABLE"), schema=os.getenv("DB_PREDICTION_SCHEMA"))

def save_predictions(data: pd.DataFrame, save_mode: str, engine: object, table: str, schema: str, if_exists: str = "replace", index: bool = False) -> None:
    """
    Save the predictions to the database.

    Args:
        data (pd.DataFrame): Data with predictions to be saved.
        save_mode (str): Mode to save the predictions. Options:
            - "sql": Save to SQL database.
            - Other options to be implemented.
        engine (Engine): Database engine. Required if save_mode is "sql".
        schema (str): Schema to save the predictions. Required if save_mode is "sql".
        if_exists (str): What to do if the table already exists in the database. Options:
            - "replace": Replace the table. Default.
            - Other options to be implemented.
        index (bool): Whether to save the index. Default is False.
    """
    if save_mode == "sql":
        data.to_sql(name=table, con=engine, schema=schema, if_exists=if_exists, index=index)
        print(f"Predictions saved to database.")
    else:
        raise ValueError("Save mode not supported.")
