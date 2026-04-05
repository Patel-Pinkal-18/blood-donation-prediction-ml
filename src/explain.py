# src/explain.py (optional)
import joblib, pandas as pd, os
import shap, numpy as np
import matplotlib.pyplot as plt

model = joblib.load("models/final_model.pkl")
df = pd.read_csv("data/cleaned_transfusion.csv")
X = df.drop(columns=["Target"])  # or appropriate column name

# If pipeline includes scaling and is sklearn Pipeline, get the final estimator:
# explainer needs either raw model or use KernelExplainer on pipeline.predict_proba
explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
shap_values = explainer.shap_values(X.iloc[:200,:])  # small sample to save time

# Save summary plot
shap.summary_plot(shap_values[1], X.iloc[:200,:], show=False)
plt.savefig("outputs/plots/shap_summary.png")
