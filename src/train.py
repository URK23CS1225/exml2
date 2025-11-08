import os
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np

# Set random seed for consistent results
np.random.seed(42)

# Directory to save model and metrics
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

def main():
    # Load a small dataset (Iris)
    X, y = load_iris(return_X_y=True)

    # Split dataset into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple machine learning model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save trained model
    dump(model, ARTIFACT_DIR / "model.joblib")

    # Calculate accuracy
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    # Save metrics into a text file
    with open(ARTIFACT_DIR / "train_metrics.txt", "w") as f:
        f.write(f"Training Accuracy: {train_acc}\n")
        f.write(f"Validation Accuracy: {val_acc}\n")

    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    main()
