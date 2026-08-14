"""Consistent evaluation utilities."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
    else:
        score = y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "Specificity": tn/(tn+fp) if tn+fp else 0.0,
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, score)
    }

def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()

def plot_roc_curves(fitted_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8,6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves")
    plt.tight_layout()
    plt.show()

def results_table(results):
    return pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
