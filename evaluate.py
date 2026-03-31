import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model      = saved["model"]
scaler     = saved["scaler"]
le         = saved["label_encoder"]
features   = saved["features"]
model_name = saved["model_name"]

df = pd.read_csv("archive/data.csv")
df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
y = le.transform(df["diagnosis"])
X = df.drop(columns="diagnosis")[features]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=48, stratify=y)

# Scale if model needs it (KNN / SVM)
needs_scaling = model_name in ("KNN", "SVM")
X_test_eval = scaler.transform(X_test) if needs_scaling else X_test
X_full_eval = scaler.transform(X)      if needs_scaling else X

y_pred = model.predict(X_test_eval)
print(f"Model        : {model_name}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)
cv_scores = cross_val_score(model, X_full_eval, y, cv=cv, scoring="accuracy")
print(f"CV Accuracy  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Per fold     : {cv_scores.round(4)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title(f"Confusion Matrix — {model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")
