# Breast Cancer Detection

A command-line tool that predicts whether a breast tumor is **Benign** or **Malignant** based on cell nucleus measurements from a biopsy report, using a machine learning model trained on the [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

---

## Project Structure

```
breast_cancer_detection/
├── archive/
│   └── data.csv            # Dataset (569 samples, 30 features)
├── bot.py                  # Interactive CLI chatbot
├── training.py             # Model training & selection
├── evaluate.py             # Evaluation & confusion matrix
├── model.pkl               # Saved best model + scaler + encoder
├── confusion_matrix.png    # Confusion matrix plot
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

If the dataset is not present, `training.py` will download it automatically via the Kaggle API. Make sure your `~/.kaggle/kaggle.json` credentials are in place.

---

## Usage

### Train
```bash
python3 training.py
```
Trains 6 classifiers (Random Forest, Gradient Boosting, XGBoost, KNN, Decision Tree, SVM), selects the best by 5-fold CV accuracy, and saves it to `model.pkl`.

### Evaluate
```bash
python3 evaluate.py
```
Prints test accuracy, classification report, and CV scores. Saves `confusion_matrix.png`.

### Run the chatbot
```bash
python3 bot.py
```
Walks you through entering up to 10 cell measurements interactively. Missing values are filled with dataset averages. Outputs a prediction with confidence.

---

## Model Performance (Gradient Boosting)

| Metric | Value |
|---|---|
| Test Accuracy | 95.80% |
| CV Accuracy (5-fold) | 96.66% ± 1.79% |
| Benign F1-score | 0.97 |
| Malignant F1-score | 0.94 |

---

## Disclaimer

This tool is for **educational and informational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional.
