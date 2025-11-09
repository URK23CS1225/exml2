# train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# Create folder to store models
Path("models").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("winequality-red.csv", sep=";")

# Convert quality into binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Train RandomForest -------- #
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
joblib.dump(rf, "models/random_forest.pkl")

# -------- Train Logistic Regression -------- #
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
joblib.dump(lr, "models/logistic_regression.pkl")

print("\n✅ Models trained and saved successfully!")
print(f"🔹 Random Forest Accuracy: {rf_acc:.2f}")
print(f"🔹 Logistic Regression Accuracy: {lr_acc:.2f}")
