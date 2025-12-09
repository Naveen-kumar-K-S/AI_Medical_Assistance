# ================================================
# ML MODEL: Disease prediction from symptoms
# ================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load raw dataset with Symptom_1..Symptom_17 text columns
print("🚀 Loading dataset...")
df_raw = pd.read_csv(r"D:\AI-Medical_Assistance\AI-Medical_Assistance\data\dataset.csv")
print("Shape of data:", df_raw.shape)
print(df_raw.head(), "\n")

# ---- Convert Symptom_1..Symptom_17 strings -> one-hot matrix ----
symptom_cols_raw = [c for c in df_raw.columns if c.startswith("Symptom_")]
disease_col = "Disease"

# collect all unique symptom names
all_symptoms = set()
for c in symptom_cols_raw:
    for val in df_raw[c].dropna().astype(str):
        s = val.strip()
        if s:
            all_symptoms.add(s)
all_symptoms = sorted(all_symptoms)

# build 0/1 feature table
X = pd.DataFrame(0, index=df_raw.index, columns=all_symptoms)
for c in symptom_cols_raw:
    for idx, val in df_raw[c].items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s in X.columns:
            X.at[idx, s] = 1

y = df_raw[disease_col]

print("One-hot feature shape:", X.shape)
print()

# Encode target as category codes
disease_labels = y.astype("category")
y_encoded = disease_labels.cat.codes

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train shape :", X_train.shape)
print("Test shape  :", X_test.shape)
print()

# 3. Train RandomForest classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

rf_clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print()

# 5. Demo prediction on a simple symptom set
symptom_cols = list(X.columns)

demo_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
demo_row = {c: 0 for c in symptom_cols}
for s in demo_symptoms:
    if s in demo_row:
        demo_row[s] = 1

demo_X = pd.DataFrame([demo_row])
demo_pred_code = rf_clf.predict(demo_X)[0]
demo_pred_disease = disease_labels.cat.categories[demo_pred_code]

print("🔍 Demo ML prediction from symptoms:", demo_symptoms)
print("Predicted disease:", demo_pred_disease)

# 6. Load precautions table and helpers
prec_df = pd.read_csv(r"D:\AI-Medical_Assistance\AI-Medical_Assistance\data\symptom_precaution.csv")

def get_precautions_for_disease(disease_name: str):
    row = prec_df[prec_df["Disease"] == disease_name]
    if row.empty:
        return []
    row = row.iloc[0]
    precs = [str(v) for c, v in row.items() if "Precaution" in c or "precaution" in c]
    return [p for p in precs if p and p != "nan"]


print("\n🎯 Demo prediction on one sample:")
sample_X = X_test.iloc[[0]]
true_code = y_test.iloc[0]
true_disease = disease_labels.cat.categories[true_code]
sample_pred_code = rf_clf.predict(sample_X)[0]
sample_pred_disease = disease_labels.cat.categories[sample_pred_code]
print("True disease :", true_disease)
print("Predicted    :", sample_pred_disease)

print("\nML model training and evaluation complete.")

# 7. Public helper used by ai-ml-combined.py

def predict_disease_from_symptoms(symptom_list):
    """
    symptom_list: list of symptom names like ["cough", "headache"].
    Uses one‑hot columns built from Symptom_1..Symptom_17.
    """
    row = {c: 0 for c in symptom_cols}
    for s in symptom_list:
        if s in row:
            row[s] = 1
    X_input = pd.DataFrame([row])
    code = rf_clf.predict(X_input)[0]
    return disease_labels.cat.categories[code]

def get_all_symptoms():
    """Return list of all canonical symptom column names used by the ML model."""
    return list(symptom_cols)
