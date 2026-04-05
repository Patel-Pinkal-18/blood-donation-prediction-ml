# src/score.py
import sys, os, joblib, pandas as pd

MODEL = os.path.join(os.getcwd(), "models", "final_model.pkl")
model = joblib.load(MODEL)

infile = sys.argv[1] if len(sys.argv) > 1 else "data/new_patients.csv"
outfile = sys.argv[2] if len(sys.argv) > 2 else "outputs/scored_new_patients.csv"

df = pd.read_csv(infile)
X = df  # ensure df has same columns as training X

proba = model.predict_proba(X)[:,1]
df["y_proba"] = proba
df["y_pred"] = (df["y_proba"] >= 0.5).astype(int)  # default threshold
df.to_csv(outfile, index=False)
print("Saved scored file to:", outfile)
