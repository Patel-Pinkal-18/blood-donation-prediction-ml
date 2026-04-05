
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tpot import TPOTClassifier
import os
import joblib

file_path = os.path.join(os.path.dirname(__file__), "..", "data", "transfusion.data")

try:
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"⚠️ File not found at {file_path}. Please check filename & location.")

print("\n--- Dataset Info ---")
print(data.info())
print("\n--- First 5 Rows ---")
print(data.head())

data.columns = ["Recency", "Frequency", "Monetary", "Time", "Target"]

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Summary Statistics ---")
print(data.describe())

sns.countplot(x="Target", data=data)
plt.title("Target Distribution")
plt.savefig("target_distribution.png")
plt.close()

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

X = data.drop("Target", axis=1)
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\n--- Random Forest ---")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\n--- Running TPOT AutoML (may take time) ---")
tpot = TPOTClassifier(
    generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1
)
tpot.fit(X_train, y_train)

print("TPOT Score on Test Set:", tpot.score(X_test, y_test))
tpot.export("best_pipeline.py")

print("\n✅ Cleaning & modeling pipeline complete! Results saved.")

output_path = os.path.join(os.path.dirname(__file__), "cleaned_transfusion.csv")
data.to_csv(output_path, index=False)
print(f"📂 Cleaned dataset saved at: {output_path}") 


os.makedirs('models', exist_ok=True)
joblib.dump(tpot.fitted_pipeline_, 'models/final_model.pkl')
print("Saved models/final_model.pkl")
