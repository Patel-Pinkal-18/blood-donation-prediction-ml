# src/evaluate_and_export.py
import os, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

ROOT = os.getcwd()
DATA = os.path.join(ROOT, "data", "cleaned_transfusion.csv")
MODEL = os.path.join(ROOT, "models", "final_model.pkl")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# Load data and model
df = pd.read_csv(DATA)
model = joblib.load(MODEL)  # assumes pipeline includes any scaling
print("Model loaded:", MODEL)

# Split to get the same test set (use same random_state as earlier)
X = df.drop(columns=[col for col in df.columns if "donated" in col.lower() or col.lower()=="target"])
# heuristics: prefer exact column name if present:
if "Target" in df.columns:
    target_col = "Target"
elif any("donated" in c.lower() for c in df.columns):
    target_col = [c for c in df.columns if "donated" in c.lower()][0]
else:
    raise ValueError("Cannot find target column. Rename target to 'target' or keep original donated column.")

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predictions & metrics
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")

# Save ROC/PR points for Tableau
fpr, tpr, thr_roc = roc_curve(y_test, y_proba)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_roc}).to_csv(os.path.join(OUT, "roc_points.csv"), index=False)

prec, rec, thr_pr = precision_recall_curve(y_test, y_proba)
# Ensure all arrays have the same length
min_length = min(len(rec[:-1]), len(prec[:-1]), len(thr_pr))
pd.DataFrame({
    "recall": rec[:-1][:min_length], 
    "precision": prec[:-1][:min_length], 
    "threshold": thr_pr[:min_length]
}).to_csv(os.path.join(OUT, "pr_points.csv"), index=False)

# Find best thresholds (Youden's J and best F1)
j_scores = tpr - fpr
best_j = thr_roc[np.argmax(j_scores)]
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thr_pr[:-1]]
best_f1 = thr_pr[:-1][int(np.argmax(f1_scores))]

print("Best threshold Youden J:", best_j)
print("Best threshold F1:", best_f1)

# Evaluate at chosen thresholds
def eval_at(th):
    pred = (y_proba >= th).astype(int)
    cm = confusion_matrix(y_test, pred)
    rep = classification_report(y_test, pred, output_dict=True)
    return {"threshold": th, "confusion_matrix": cm, "report": rep}

metrics_default = eval_at(0.5)
metrics_j = eval_at(best_j)
metrics_f1 = eval_at(best_f1)

# Save metrics row
metrics_row = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "threshold_default": 0.5,
    "threshold_j": float(best_j),
    "threshold_f1": float(best_f1),
    "precision_at_j": precision_score(y_test, (y_proba>=best_j).astype(int)),
    "recall_at_j": recall_score(y_test, (y_proba>=best_j).astype(int)),
    "f1_at_j": f1_score(y_test, (y_proba>=best_j).astype(int))
}
pd.DataFrame([metrics_row]).to_csv(os.path.join(OUT, "tableau_metrics.csv"), index=False)

# Build predicted table for Tableau
pred_table = X_test.reset_index(drop=True).copy()
pred_table["y_true"] = y_test.reset_index(drop=True)
pred_table["y_proba"] = y_proba
pred_table["y_pred_default"] = (y_proba >= 0.5).astype(int)
pred_table["y_pred_bestf1"] = (y_proba >= best_f1).astype(int)
pred_table.to_csv(os.path.join(OUT, "tableau_predictions.csv"), index=False)

# Save confusion matrices & plots
import seaborn as sns
os.makedirs(os.path.join(OUT,"plots"), exist_ok=True)

def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.title(name)
    plt.savefig(os.path.join(OUT,"plots", f"{name.replace(' ','_')}.png"))
    plt.close()

plot_confusion(metrics_default["confusion_matrix"], "confusion_default")
plot_confusion(metrics_f1["confusion_matrix"], "confusion_bestf1")

# ROC & PR plots
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.savefig(os.path.join(OUT,"plots","roc_curve.png")); plt.close()

plt.figure()
plt.plot(rec[:-1], prec[:-1], label=f'PR AUC {pr_auc:.3f}')
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
plt.legend(); plt.savefig(os.path.join(OUT,"plots","pr_curve.png")); plt.close()

# Permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring='roc_auc')
imp_df = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

print("Outputs written to:", OUT)
