import pandas as pd
import os

def load_data():
    path = "data/raw/churn.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if os.stat(path).st_size == 0:
        raise ValueError("File is empty!")

    df = pd.read_csv(path)

    print("Data Loaded Successfully. Shape:", df.shape)

    return df