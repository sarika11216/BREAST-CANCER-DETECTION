import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

dataset  = "uciml/breast-cancer-wisconsin-data"
data_dir = "archive"
data_file = os.path.join(data_dir, "data.csv")

if not os.path.exists(data_file):
    print("Downloading dataset from Kaggle...")
    os.makedirs(data_dir, exist_ok=True)
    kaggle_bin = os.path.expanduser("~/Library/Python/3.14/bin/kaggle")
    subprocess.run(
        [kaggle_bin, "datasets", "download", "-d", dataset, "--unzip", "-p", data_dir],
        check=True
    )
    print("Download complete.\n")

df = pd.read_csv(data_file)
print(f"Raw shape      : {df.shape}")

df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")

df.dropna(how="all", inplace=True)

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed : {before - len(df)}")

feature_cols = [c for c in df.columns if c != "diagnosis"]
for col in feature_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    df[col] = df[col].clip(lower, upper)

print(f"Clean shape    : {df.shape}")
print(f"Class balance  :\n{df['diagnosis'].value_counts()}\n")

le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])
X = df.drop(columns="diagnosis")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=48, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300, random_state=48, n_jobs=-1,
            criterion="entropy", min_samples_split=5, min_samples_leaf=2,
            max_depth=5, max_features="sqrt", bootstrap=True,
            class_weight="balanced", oob_score=True
        ),
        False
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(
            n_estimators=300, random_state=48, learning_rate=0.15,
            max_depth=5, max_features="sqrt", subsample=0.8,
            min_samples_split=5, min_samples_leaf=2
        ),
        False
    ),
    "XGBoost": (
        XGBClassifier(
            n_estimators=300, random_state=48, learning_rate=0.4,
            max_depth=5, n_jobs=-1, gamma=0.1, eval_metric="logloss"
        ),
        False
    ),
    "KNN": (
        KNeighborsClassifier(
            n_neighbors=8, weights="uniform", metric="minkowski", n_jobs=-1
        ),
        True   # needs scaling
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=48, criterion="entropy", min_samples_split=5,
            min_samples_leaf=2, max_depth=5, max_features="sqrt",
            class_weight="balanced"
        ),
        False
    ),
    "SVM": (
        SVC(
            C=10, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=True, random_state=48
        ),
        True
    ),
}

best_model, best_name, best_acc = None, "", 0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)

for name, (clf, use_scaled) in models.items():
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test
    X_full = scaler.transform(X) if use_scaled else X

    cv_scores = cross_val_score(clf, X_full, y, cv=cv, scoring="accuracy")
    clf.fit(Xtr, y_train)
    acc = accuracy_score(y_test, clf.predict(Xte))

    print(f"{name}")
    print(f"  CV Accuracy  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {acc:.2%}\n")

    if cv_scores.mean() > best_acc:
        best_acc, best_name, best_model = cv_scores.mean(), name, clf

print(f"Best: {best_name} ({best_acc:.2%})")

with open("model.pkl", "wb") as f:
    pickle.dump({
        "model":          best_model,
        "scaler":         scaler,
        "label_encoder":  le,
        "features":       list(X.columns),
        "model_name":     best_name
    }, f)

print("Saved to model.pkl")
