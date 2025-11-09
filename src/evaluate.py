import json
from pathlib import Path
from joblib import load
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")

def main():
    model = load(ARTIFACTS_DIR / "model.joblib")
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Validation accuracy: {val_acc:.4f}")

    # (Optionally) overwrite metrics.json with latest val acc
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except FileNotFoundError:
        metrics = {}
    metrics["val_accuracy"] = round(val_acc, 4)
    metrics_path.write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
