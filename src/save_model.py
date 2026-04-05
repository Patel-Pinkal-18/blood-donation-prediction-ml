# src/save_model.py
import os, joblib

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# If cleaning.py left a pickle inside models, skip. Otherwise, import or re-create.
# Preferred: you ran cleaning.py and it placed tpot.fitted_pipeline_ in-memory - if not, re-run cleaning.py first.

# Example: if cleaning.py saved `tpot_pipeline.pkl`:
SOURCE_PATH = os.path.join(os.getcwd(), "models", "tpot_pipeline_temp.pkl")  # replace if used
FINAL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")

# If you already have the variable 'tpot' in memory, you can directly joblib.dump(tpot.fitted_pipeline_, FINAL_PATH)
# But typically run this at the end of your cleaning.py to save directly. Example snippet to add at end of cleaning.py:
"""
import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump(tpot.fitted_pipeline_, 'models/final_model.pkl')
print('Saved models/final_model.pkl')
"""
# If the file already exists:
if os.path.exists(FINAL_PATH):
    print(f"{FINAL_PATH} already exists.")
else:
    raise FileNotFoundError("final_model.pkl not found. Re-run cleaning.py with a save step to create models/final_model.pkl")
