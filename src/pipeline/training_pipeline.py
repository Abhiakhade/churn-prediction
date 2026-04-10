from src.data.ingestion import load_data
from src.features.engineering import clean_data
from src.features.preprocessing import build_preprocessor
from src.models.train import train_model

from sklearn.model_selection import train_test_split


def run_training_pipeline():
    # -----------------------------
    # Load Data
    # -----------------------------
    df = load_data()

    # -----------------------------
    # Clean Data
    # -----------------------------
    df = clean_data(df)

    # -----------------------------
    # Split Features / Target
    # -----------------------------
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # Column Types
    # -----------------------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # -----------------------------
    # Preprocessing
    # -----------------------------
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Train Model
    # -----------------------------
    model = train_model(X_train, y_train, X_test, y_test, preprocessor)

    print("\n🚀 Training pipeline completed successfully!")