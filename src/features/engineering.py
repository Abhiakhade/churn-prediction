import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove customerID (not useful)
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values
    df = df.dropna()

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df