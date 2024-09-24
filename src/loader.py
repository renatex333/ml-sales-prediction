import os
import pickle
import pandas as pd
from sqlalchemy import create_engine

def load_data(path: str, source_type: str, db_url: str) -> pd.DataFrame:
    """
    Load data from the database.

    Args:
        path (str): Path to the data.
        source_type (str): From where the data will be loaded. Options:
            - "sql": SQL script to retrieve the data from the database.
            - Other options to be implemented.
        db_url (str): Database URL. If source_type is "sql", this argument is required.

    Returns:
        pd.DataFrame: Data loaded from the database.
    """
    if source_type == "sql":
        return load_data_from_sql(path, db_url)
    else:
        raise ValueError("Source type not supported.")

def load_data_from_sql(query_path: str, db_url: str) -> pd.DataFrame:
    """
    Load data from a SQL script.

    Args:
        query_path (str): Path to the SQL script.
        db_url (str): Database URL.

    Returns:
        pd.DataFrame: Data loaded from the database.
    """
    if db_url is None:
        raise ValueError("Database URL is required.")
    if query_path is None:
        raise ValueError("Query path is required.")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"File not found: {query_path}")

    engine = create_engine(db_url)
    with open(query_path, "r") as f:
        sql_script = f.read()
    with engine.connect() as conn:
        data = pd.read_sql(
            sql=sql_script,
            con=conn.connection
        )
    return data

def load_model(path: str) -> object:
    """
    Load the model from disk.

    Args:
        path (str): Path to the model. Can be a path to:
            - Pickle file.
            - Other options to be implemented.

    Returns:
        Model loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".pkl"):
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
    else:
        raise ValueError("Model format not supported.")
