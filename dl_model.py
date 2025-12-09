# ================================================
# DL MODEL: MLP disease classifier
# ================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("🚀 Loading dataset for DL...")
df_raw = pd.read_csv(r"D:\AI-Medical_Assistance\AI-Medical_Assistance\data\dataset.csv")

symptom_cols_raw = [c for c in df_raw.columns if c.startswith("Symptom_")]
disease_col = "Disease"

# same one-hot encoding as ml_model.py
all_symptoms = set()
for c in symptom_cols_raw:
    for val in df_raw[c].dropna().astype(str):
        s = val.strip()
        if s:
            all_symptoms.add(s)
all_symptoms = sorted(all_symptoms)

X = pd.DataFrame(0, index=df_raw.index, columns=all_symptoms)
for c in symptom_cols_raw:
    for idx, val in df_raw[c].items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s in X.columns:
            X.at[idx, s] = 1

y = df_raw[disease_col]

disease_labels = y.astype("category")
y_encoded = disease_labels.cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train shape :", X_train.shape)
print("Test shape  :", X_test.shape)
print()

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ DL Accuracy:", acc)
print("\n✅ DL Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ DL Classification Report:")
print(classification_report(y_test, y_pred))

# simple demo
sample_X = X_test.iloc[[0]]
true_code = y_test.iloc[0]
true_disease = disease_labels.cat.categories[true_code]
pred_code = mlp.predict(sample_X)[0]
pred_disease = disease_labels.cat.categories[pred_code]

print("\n🎯 DL Demo prediction on one sample:")
print("True disease :", true_disease)
print("Predicted    :", pred_disease)
print("\nDL model training and evaluation complete.")