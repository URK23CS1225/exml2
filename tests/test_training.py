from pathlib import Path
import json
import subprocess
import sys

def run(cmd: str):
    result = subprocess.run(cmd, shell=True, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout

def test_train_produces_artifacts(tmp_path, monkeypatch):
    # run training in project root (artifacts folder is fixed)
    out = run(f"{sys.executable} src/train.py")
    assert "Saved model" in out

    model_path = Path("artifacts/model.joblib")
    metrics_path = Path("artifacts/metrics.json")
    assert model_path.exists(), "model.joblib was not created"
    assert metrics_path.exists(), "metrics.json was not created"

    metrics = json.loads(metrics_path.read_text())
    assert metrics["val_accuracy"] >= 0.8  # sanity check threshold
