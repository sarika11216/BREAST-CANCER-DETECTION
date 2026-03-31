import warnings
warnings.filterwarnings("ignore")

import pickle
import time
import numpy as np
import pandas as pd

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model      = saved["model"]
scaler     = saved["scaler"]
le         = saved["label_encoder"]
features   = saved["features"]
model_name = saved["model_name"]

df_ref = pd.read_csv("archive/data.csv")
df_ref.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
df_ref = df_ref.drop(columns="diagnosis")
stats = df_ref.describe()

KEY_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

FEATURE_HINTS = {
    "radius_mean":            "mean radius of the cell nuclei (typical range 6–28)",
    "texture_mean":           "mean texture — std dev of gray-scale values (typical 9–40)",
    "perimeter_mean":         "mean perimeter of nuclei (typical 43–190)",
    "area_mean":              "mean area of nuclei (typical 143–2501)",
    "smoothness_mean":        "mean smoothness — local variation in radius (typical 0.05–0.16)",
    "compactness_mean":       "mean compactness (perimeter²/area − 1, typical 0.02–0.35)",
    "concavity_mean":         "mean severity of concave portions (typical 0–0.43)",
    "concave points_mean":    "mean number of concave portions (typical 0–0.20)",
    "symmetry_mean":          "mean symmetry of nuclei (typical 0.10–0.30)",
    "fractal_dimension_mean": "mean fractal dimension (typical 0.05–0.10)",
}


def slow_print(text, delay=0.018):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def ask_float(prompt, feature):
    lo = stats.loc["min", feature]
    hi = stats.loc["max", feature]
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            val = float(raw)
            if not (lo * 0.5 <= val <= hi * 2):
                print(f"  ⚠  That looks unusual (dataset range: {lo:.3g}–{hi:.3g}). Re-enter or press Enter to skip.")
                continue
            return val
        except ValueError:
            print("  Please enter a numeric value.")


def predict(values: dict):
    row = {f: values.get(f, stats.loc["mean", f]) for f in features}
    X = pd.DataFrame([row])
    needs_scaling = model_name in ("KNN", "SVM")
    X_eval = scaler.transform(X) if needs_scaling else X
    proba = model.predict_proba(X_eval)[0]
    classes = le.classes_
    top = sorted(zip(classes, proba), key=lambda x: -x[1])
    return top


def interpret(label, conf):
    if label == "M":
        if conf >= 0.85:
            return "High likelihood of malignancy. Please see a specialist urgently."
        else:
            return "Possible malignancy — confidence is moderate. A follow-up is strongly advised."
    else:
        if conf >= 0.85:
            return "Looks benign with high confidence. Routine monitoring is still recommended."
        else:
            return "Likely benign, but confidence is moderate. A check-up would be wise."


def chat():
    while True:
        print("\n" + "=" * 56)
        slow_print("      Breast Cancer Detection Assistant")
        print("=" * 56 + "\n")
        slow_print("Hi there! I'll walk you through entering a few cell")
        slow_print("measurements from a biopsy report. I'll use those to")
        slow_print("predict whether the tumor is Benign (B) or Malignant (M).\n")
        slow_print("You can skip any value by pressing Enter — I'll fill in")
        slow_print("the dataset average for that field.\n")

        values = {}
        for feat in KEY_FEATURES:
            hint = FEATURE_HINTS[feat]
            label = feat.replace("_", " ").title()
            val = ask_float(f"  {label}\n  ({hint})\n  → ", feat)
            if val is not None:
                values[feat] = val
                print(f"  ✓ Recorded: {val}\n")
            else:
                print(f"  → Using dataset average ({stats.loc['mean', feat]:.4g})\n")

        print("  Analyzing", end="", flush=True)
        for _ in range(5):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print("\n")

        top = predict(values)
        label, conf = top[0]
        full_label = "Malignant" if label == "M" else "Benign"

        print("=" * 56)
        print(f"  Prediction  : {full_label} ({label})")
        print(f"  Confidence  : {conf:.1%}")
        print(f"  Model used  : {model_name}")
        print(f"  Features    : {len(values)} entered, {len(features) - len(values)} auto-filled")
        print("=" * 56)

        if len(top) > 1 and top[1][1] > 0.05:
            print(f"\n  Other possibility:")
            d, p = top[1]
            full_d = "Malignant" if d == "M" else "Benign"
            print(f"    - {full_d} ({p:.1%})")

        print()
        slow_print(f"  {interpret(label, conf)}")

        if len(values) < len(KEY_FEATURES) // 2:
            print()
            slow_print("  Tip: You entered fewer than half the fields.")
            slow_print("  More data = more accurate prediction.")

        print()
        slow_print("  ⚠  This tool is for informational purposes only.")
        slow_print("     Always consult a qualified medical professional.")

        print()
        again = input("  Run another prediction? (yes/no) ").strip().lower()
        if again not in ("yes", "y"):
            print()
            slow_print("  Take care of yourself. Goodbye!")
            print()
            break


chat()
