"""
evaluation.py
-------------
Visual evaluation utilities.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "loan_model.pkl")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_model():
    return joblib.load(MODEL_PATH)


def plot_confusion_matrix(model, X_test, y_test):
    os.makedirs(FIG_DIR, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("📊 Saved confusion_matrix.png")


def plot_roc_curve(model, X_test, y_test):
    os.makedirs(FIG_DIR, exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("📈 Saved roc_curve.png")


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Model has no feature_importances_")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    importances = pd.Series(model.feature_importances_,
                            index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("📉 Saved feature_importance.png")


def main():
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    model = load_model()
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns)


if __name__ == "__main__":
    main()