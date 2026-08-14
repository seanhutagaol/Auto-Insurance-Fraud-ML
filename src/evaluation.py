"""Evaluation utilities for binary insurance-fraud classification."""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)


def evaluate_model_performance(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    plot: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """Evaluate a fitted classifier."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        raise AttributeError(
            "Model must provide predict_proba or decision_function."
        )

    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"=== Performance Report: {model_name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC:     {roc_auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].imshow(cm)
        axes[0].set_title(f"Confusion Matrix: {model_name}")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")
        axes[0].set_xticks([0, 1], ["Non-Fraud", "Fraud"])
        axes[0].set_yticks([0, 1], ["Non-Fraud", "Fraud"])

        for i in range(2):
            for j in range(2):
                axes[0].text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=axes[1],
            name=model_name,
        )
        axes[1].set_title(f"ROC Curve: {model_name}")

        plt.tight_layout()
        plt.show()

    return cm, roc_auc, specificity, sensitivity


def compare_models(results: dict):
    """Return a model-comparison DataFrame sorted by ROC AUC."""
    import pandas as pd

    return (
        pd.DataFrame(results).T
        .sort_values("ROC AUC", ascending=False)
    )
