import os
import sqlite3
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# =========================
# 1. CREATE FOLDERS
# =========================
BASE_DIR = os.getcwd()

DATA_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================
# 2. LOAD DATA
# =========================
# Put your file in data folder with name: blood.csv
DATA_FILE = os.path.join(DATA_DIR, "blood.csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"File not found: {DATA_FILE}\n"
        f"Please place your blood.csv file inside the data folder."
    )

df = pd.read_csv(DATA_FILE)
print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")


# =========================
# 3. RENAME COLUMNS
# =========================
# Adjust this if your dataset already has proper names
if df.shape[1] == 5:
    df.columns = ["Recency", "Frequency", "Monetary", "Time", "Target"]

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())


# =========================
# 4. SAVE CLEANED DATA
# =========================
CLEANED_FILE = os.path.join(DATA_DIR, "cleaned_transfusion.csv")
df.to_csv(CLEANED_FILE, index=False)
print(f"\n✅ Cleaned dataset saved at: {CLEANED_FILE}")


# =========================
# 5. BASIC EDA PLOTS
# =========================
plt.figure(figsize=(6, 4))
sns.countplot(x="Target", data=df)
plt.title("Target Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "target_distribution.png"))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
plt.close()

print("✅ EDA plots saved.")


# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n✅ Train-test split completed.")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# =========================
# 7. BUILD MODELS
# =========================
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# Train models
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("\n✅ Models trained successfully.")


# =========================
# 8. MODEL PREDICTIONS
# =========================
log_pred = log_model.predict(X_test)
log_proba = log_model.predict_proba(X_test)[:, 1]

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]


# =========================
# 9. EVALUATE MODELS
# =========================
def evaluate_model(name, y_true, y_pred, y_proba):
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- {name} ---")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(roc_auc, 4))
    print("PR AUC   :", round(pr_auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

log_metrics = evaluate_model("Logistic Regression", y_test, log_pred, log_proba)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)


# =========================
# 10. SAVE METRICS CSV
# =========================
metrics_df = pd.DataFrame([log_metrics, rf_metrics])
metrics_csv = os.path.join(OUTPUT_DIR, "model_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"✅ Metrics saved at: {metrics_csv}")


# =========================
# 11. ROC CURVE
# =========================
log_fpr, log_tpr, log_thr = roc_curve(y_test, log_proba)
rf_fpr, rf_tpr, rf_thr = roc_curve(y_test, rf_proba)

plt.figure(figsize=(7, 5))
plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_metrics['roc_auc']:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))
plt.close()

roc_points = pd.DataFrame({
    "fpr": rf_fpr,
    "tpr": rf_tpr,
    "threshold": rf_thr
})
roc_points.to_csv(os.path.join(OUTPUT_DIR, "roc_points.csv"), index=False)


# =========================
# 12. PR CURVE
# =========================
rf_precision, rf_recall, rf_pr_thr = precision_recall_curve(y_test, rf_proba)

plt.figure(figsize=(7, 5))
plt.plot(rf_recall[:-1], rf_precision[:-1], label=f"Random Forest (PR AUC = {rf_metrics['pr_auc']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pr_curve.png"))
plt.close()

min_len = min(len(rf_recall[:-1]), len(rf_precision[:-1]), len(rf_pr_thr))
pr_points = pd.DataFrame({
    "recall": rf_recall[:-1][:min_len],
    "precision": rf_precision[:-1][:min_len],
    "threshold": rf_pr_thr[:min_len]
})
pr_points.to_csv(os.path.join(OUTPUT_DIR, "pr_points.csv"), index=False)

print("✅ ROC and PR outputs saved.")


# =========================
# 13. THRESHOLD OPTIMIZATION
# =========================
j_scores = rf_tpr - rf_fpr
best_j_threshold = rf_thr[np.argmax(j_scores)]

f1_scores = [f1_score(y_test, (rf_proba >= t).astype(int), zero_division=0) for t in rf_pr_thr]
best_f1_threshold = rf_pr_thr[int(np.argmax(f1_scores))]

print("\nBest Threshold by Youden's J:", round(float(best_j_threshold), 4))
print("Best Threshold by F1 Score  :", round(float(best_f1_threshold), 4))


# =========================
# 14. CONFUSION MATRICES
# =========================
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    return cm

cm_default = save_confusion_matrix(
    y_test,
    (rf_proba >= 0.5).astype(int),
    "Confusion Matrix - Default Threshold",
    "confusion_default.png"
)

cm_bestf1 = save_confusion_matrix(
    y_test,
    (rf_proba >= best_f1_threshold).astype(int),
    "Confusion Matrix - Best F1 Threshold",
    "confusion_bestf1.png"
)

print("✅ Confusion matrices saved.")


# =========================
# 15. FEATURE IMPORTANCE
# =========================
perm_result = permutation_importance(
    rf_model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="roc_auc"
)

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm_result.importances_mean
}).sort_values(by="importance", ascending=False)

feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance_df, x="importance", y="feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
plt.close()

print("✅ Feature importance saved.")


# =========================
# 16. PREDICTION TABLE
# =========================
pred_table = X_test.reset_index(drop=True).copy()
pred_table["y_true"] = y_test.reset_index(drop=True)
pred_table["y_proba_logistic"] = log_proba
pred_table["y_pred_logistic"] = log_pred
pred_table["y_proba_rf"] = rf_proba
pred_table["y_pred_rf_default"] = (rf_proba >= 0.5).astype(int)
pred_table["y_pred_rf_bestf1"] = (rf_proba >= best_f1_threshold).astype(int)

pred_csv = os.path.join(OUTPUT_DIR, "tableau_predictions.csv")
pred_table.to_csv(pred_csv, index=False)

tableau_metrics = pd.DataFrame([{
    "roc_auc_rf": rf_metrics["roc_auc"],
    "pr_auc_rf": rf_metrics["pr_auc"],
    "threshold_default": 0.5,
    "threshold_best_j": float(best_j_threshold),
    "threshold_best_f1": float(best_f1_threshold),
    "precision_best_f1": precision_score(y_test, (rf_proba >= best_f1_threshold).astype(int), zero_division=0),
    "recall_best_f1": recall_score(y_test, (rf_proba >= best_f1_threshold).astype(int), zero_division=0),
    "f1_best_f1": f1_score(y_test, (rf_proba >= best_f1_threshold).astype(int), zero_division=0)
}])

tableau_metrics.to_csv(os.path.join(OUTPUT_DIR, "tableau_metrics.csv"), index=False)

print("✅ Prediction files saved.")


# =========================
# 17. SAVE MODEL
# =========================
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
joblib.dump(rf_model, MODEL_PATH)

PIPELINE_PATH = os.path.join(MODEL_DIR, "logistic_pipeline.pkl")
joblib.dump(log_model, PIPELINE_PATH)

print(f"✅ Random Forest model saved at: {MODEL_PATH}")
print(f"✅ Logistic pipeline saved at: {PIPELINE_PATH}")


# =========================
# 18. EXPORT TO SQLITE
# =========================
db_path = os.path.join(OUTPUT_DIR, "blood_donation.db")
conn = sqlite3.connect(db_path)

pred_table.to_sql("predictions", conn, if_exists="replace", index=False)
metrics_df.to_sql("model_metrics", conn, if_exists="replace", index=False)
feature_importance_df.to_sql("feature_importance", conn, if_exists="replace", index=False)

conn.close()
print(f"✅ SQLite database created at: {db_path}")


# =========================
# 19. OPTIONAL SCORING FILE
# =========================
def score_new_data(input_csv, output_csv=None):
    """
    Score new patient/donor data using saved Random Forest model.
    """
    model = joblib.load(MODEL_PATH)
    new_df = pd.read_csv(input_csv)

    probs = model.predict_proba(new_df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result_df = new_df.copy()
    result_df["y_proba"] = probs
    result_df["y_pred"] = preds

    if output_csv is None:
        output_csv = os.path.join(OUTPUT_DIR, "scored_new_patients.csv")

    result_df.to_csv(output_csv, index=False)
    print(f"✅ New data scored and saved at: {output_csv}")
    return result_df


# =========================
# 20. FINAL MESSAGE
# =========================
print("\n" + "=" * 60)
print("🎉 ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Cleaned data       : {CLEANED_FILE}")
print(f"Models saved       : {MODEL_DIR}")
print(f"Plots saved        : {PLOT_DIR}")
print(f"CSV outputs saved  : {OUTPUT_DIR}")
print(f"SQLite DB saved    : {db_path}")
print("=" * 60)