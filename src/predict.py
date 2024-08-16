import os
import sys
import pickle
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sklearn.ensemble import RandomForestRegressor

def main(data_path: str, model_path: str) -> None:
    """
    Main function to predict the total sales.

    Args:
        data_path (str): Path to the sql script to retrieve the data.
        model_path (str): Path to the model.
    """
    load_dotenv()
    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    with open(data_path, "r") as f:
        sql_script = f.read()
    data = pd.read_sql(sql_script, engine)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    prediction_path = data_path.split("/")[-1].replace("predict-", "predict-done-")
    if not prediction_path.startswith("predict-done-"):
        prediction_path = f"predict-done-{prediction_path}"
    predict(data, model, engine)

def predict(data: pd.DataFrame, model: RandomForestRegressor, engine: Engine) -> None:
    """
    Predict the total sales.

    Args:
        data (pd.DataFrame): Data to be used for prediction.
        model: Model to be used for prediction.
        engine (Engine): Database engine.
    """
    predictions = model.predict(data)
    data["prediction_total_sales"] = predictions

    data.to_sql(name=os.getenv("DB_PREDICTION_TABLE"), con=engine, schema=os.getenv("DB_PREDICTION_SCHEMA"), if_exists="replace", index=False)
    print(f"Predictions saved to database.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python predict.py <path to model> <path to sql script>")
        sys.exit(1)
    else:
        data_path = sys.argv[-1]
        model_path = sys.argv[-2]
        main(data_path, model_path)
        sys.exit(0)