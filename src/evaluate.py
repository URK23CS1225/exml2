from pathlib import Path
from joblib import load
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Set random seed for consistent results
np.random.seed(42)

ARTIFACT_DIR = Path("artifacts")

def main():
    # Load model saved during training
    model_path = ARTIFACT_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("❌ Model file not found. Run train.py first.")

    model = load(model_path)

    # Load dataset again for evaluation
    X, y = load_iris(return_X_y=True)

    # Predict
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    # Save evaluation results
    with open(ARTIFACT_DIR / "eval_metrics.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("✅ Evaluation completed. Metrics saved!")

if __name__ == "__main__":
    main()
