from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate permutation feature importance
result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)

# Display results
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance Mean": result.importances_mean,
    "Importance Std": result.importances_std
}).sort_values(by="Importance Mean", ascending=False)

print("\n✅ Permutation Feature Importance Results:")
print(importance_df)

# Optional: Save results for screenshot
importance_df.to_csv("feature_importance_results.csv", index=False)
import matplotlib.pyplot as plt

plt.barh(importance_df["Feature"], importance_df["Importance Mean"])
plt.xlabel("Importance Score (Decrease in Accuracy)")
plt.title("Permutation Feature Importance - Iris Dataset")
plt.show()

