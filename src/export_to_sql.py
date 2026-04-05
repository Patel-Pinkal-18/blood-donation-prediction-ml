# src/export_to_sql.py
import sqlite3, os, pandas as pd
OUT = os.path.join(os.getcwd(), "outputs")
df = pd.read_csv(os.path.join(OUT, "tableau_predictions.csv"))
conn = sqlite3.connect(os.path.join(OUT, "blood.db"))
df.to_sql("predictions", conn, if_exists="replace", index=False)
print("Wrote predictions table to outputs/blood.db")
conn.close()
