import pandas as pd
import joblib

print("Loading massive CSV...")
df = pd.read_csv("cars_cleaned.csv").dropna()

cat_features = ["make", "model", "trim", "body", "transmission", "state", "color", "interior", "seller"]
freq_dicts = {}

print("Calculating frequencies...")
for col in cat_features:
    freq_dicts[col] = df[col].astype(str).str.lower().value_counts().to_dict()

# Save the dictionaries as a tiny .pkl file!
joblib.dump(freq_dicts, "freq_dicts.pkl")