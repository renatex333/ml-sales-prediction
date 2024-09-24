import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from src.loader import load_data
from sklearn.ensemble import RandomForestRegressor

def train(data_path: str, target: str = "total_sales", model_name: str = "model", save_mode: str = "pkl") -> None:
    """
    Main function to wrap model training.

    Args:
        data_path (str): Path to the data. Can be a path to:
            - SQL script;
            - Other options to be implemented.
        target (str): Target column name.
        model_name (str): Name of the model.
        save_mode (str): Mode to save the model. Options:
            - pkl: Pickle file. Default.
            - Other options to be implemented.
    """
    load_dotenv()
    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    data = load_data(data_path, "sql", db_url)
    train_model(data, target, model_name, save_mode)

def train_model(data: pd.DataFrame, target: str, model_name: str = "model", save_mode: str = "pkl") -> None:
    """
    Train the model.

    Args:
        data (pd.DataFrame): Data to be used for training.
        target (str): Target column.
        model_name (str): Name of the model.
        save_mode (str): Mode to save the model. Options:
            - pkl: Pickle file. Default.
            - Other options to be implemented.
    """

    X = data.drop(columns=[target])
    y = data[target]
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=195)
    model.fit(X, y)

    save_model(model, model_name, save_mode)

def save_model(model: object, model_name: str = "model", save_mode: str = "pkl") -> None:
    """
    Save the model to disk.

    Args:
        model: Model to be saved.
        model_name: Name of the model.
        save_mode: Mode to save the model. Options:
            - pkl: Pickle file. Default.
            - Other options to be implemented.
    """
    model_dir = os.path.relpath("models", os.getcwd())
    model_path = os.path.join(model_dir, model_name)
    if save_mode == "pkl":
        model_path = f"{model_path}.pkl"
        with open(model_path, "wb") as f:
            print(f"Saving model to {model_path}...")
            pickle.dump(model, f)
