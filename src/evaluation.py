"""Evaluation Module.

This module provides metrics calculation, confusion matrix analysis, 
and repeated cross-validation evaluation loops to test model robustness.
"""

from typing import Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model_performance(model, X_test, y_test, model_name: str = "Model") -> Tuple[np.ndarray, float, float, float]:
    """Evaluate a trained model on test data and print comprehensive metrics.

    Parameters
    ----------
    model : estimator object
        A fitted pipeline or model object.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing labels.
    model_name : str, default="Model"
        Name of the model for visualization labeling.

    Returns
    -------
    Tuple[np.ndarray, float, float, float]
        Confusion matrix, ROC AUC score, Specificity, and Sensitivity.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"=== Performance Report: {model_name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}\n")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return cm, roc_auc, specificity, sensitivity


def run_repeated_cross_validation(pipeline, X_train, y_train, X_test, y_test, iterations: int = 5):
    """Run multiple iterations of training and evaluation to verify stability.

    Parameters
    ----------
    pipeline : estimator object
        The model pipeline to evaluate.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_test, y_test : pd.DataFrame, pd.Series
        Testing data.
    iterations : int, default=5
        Number of repeat runs.
    """
    roc_auc_scores = []
    specificity_scores = []
    sensitivity_scores = []

    for i in range(iterations):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
        else:
            probabilities = pipeline.decision_function(X_test)

        roc_auc = roc_auc_score(y_test, probabilities)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        roc_auc_scores.append(roc_auc)
        specificity_scores.append(specificity)
        sensitivity_scores.append(sensitivity)

        print(f"Iteration {i + 1} | ROC AUC: {roc_auc:.4f} | Spec: {specificity:.4f} | Sens: {sensitivity:.4f}")

    print("\n--- Summary of Repeated Cross-Validation ---")
    print(f"Average ROC AUC: {np.mean(roc_auc_scores):.4f}")
    print(f"Average Specificity: {np.mean(specificity_scores):.4f}")
    print(f"Average Sensitivity: {np.mean(sensitivity_scores):.4f}")
