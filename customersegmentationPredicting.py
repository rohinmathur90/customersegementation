import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = "C:/Users/91996/Downloads/Rohin/CustomerSegmentation.pkl"
INPUT_FILE = "C:/Users/91996/Downloads/Rohin/input.csv"
OUTPUT_FILE = "C:/Users/91996/Downloads/Rohin/output.csv"

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load(MODEL_PATH)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(INPUT_FILE)

# -------------------------------
# SAME PREPROCESSING AS TRAINING
# -------------------------------

# Drop ID column
if "CUST_ID" in df.columns:
    df = df.drop("CUST_ID", axis=1)

# Drop MINIMUM_PAYMENTS (you removed it in training)
if "MINIMUM_PAYMENTS" in df.columns:
    df = df.drop("MINIMUM_PAYMENTS", axis=1)

# Fill missing values (same logic)
if "CREDIT_LIMIT" in df.columns:
    df = df.dropna(subset=["CREDIT_LIMIT"])

# Fill remaining missing values with median
df = df.fillna(df.median(numeric_only=True))

# Log transform (VERY IMPORTANT)
for col in df.columns:
    df[col] = np.log1p(df[col])

# -------------------------------
# SCALE DATA (recreate scaler)
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)   # ⚠️ Not ideal but needed since scaler not saved

# -------------------------------
# PREDICT
# -------------------------------
clusters = model.predict(df_scaled)

# -------------------------------
# OUTPUT
# -------------------------------
df_original = pd.read_csv(INPUT_FILE)

df_original["Cluster"] = clusters

df_original.to_csv(OUTPUT_FILE, index=False)

print("✅ Prediction completed successfully!")
print(df_original.head())