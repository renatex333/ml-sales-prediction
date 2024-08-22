import os
import sys
import pickle
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str) -> None:
    """
    Main function to train the model.

    Args:
        data_path (str): Path to the sql script to retrieve the data.
    """
    load_dotenv()
    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    with open(data_path, "r") as f:
        sql_script = f.read()
    with engine.connect() as conn:
        data = pd.read_sql(
            sql=sql_script,
            con=conn.connection
        )
    train_model(data, "total_sales", data_path)

def train_model(data: pd.DataFrame, target: str, data_path: str) -> None:
    X = data.drop(columns=[target])
    y = data[target]
    
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=195)
    model.fit(X, y)

    model_name = data_path.split("/")[-1]
    for ext in ["parquet", "csv", "sql"]:
        if model_name.endswith(ext):
            model_name = model_name.replace(f".{ext}", ".pkl").replace("train-", "model-")
            break
    if not model_name.startswith("model-"):
        model_name = f"model-{model_name}"
    save_model(model, model_name)

def save_model(model: RandomForestRegressor, model_name: str) -> None:
    """
    Save the model to disk.

    Args:
        model: Model to be saved.
        model_name: Name of the model.
    """
    model_dir = os.path.relpath("models", os.getcwd())
    model_path = os.path.join(model_dir, model_name)
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python train.py <path to sql script>")
        sys.exit(1)
    else:
        data_path = sys.argv[-1]
        main(data_path)
        sys.exit(0)