import json
from pathlib import Path
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=300, n_jobs=None)
    model.fit(X_train, y_train)

    # Save model
    model_path = ARTIFACTS_DIR / "model.joblib"
    dump(model, model_path)

    # Evaluate on train/val
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    metrics = {
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Metrics -> {metrics}")

if __name__ == "__main__":
    main()
