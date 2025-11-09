# CI/CD for ML with GitHub Actions

This repo demonstrates a minimal ML pipeline (Iris dataset) with:
- Training (`src/train.py`)
- Evaluation (`src/evaluate.py`)
- Tests (`tests/test_training.py`)
- CI/CD workflow (`.github/workflows/ci-cd.yml`) that installs deps, runs tests, trains, evaluates, and uploads artifacts.
