"""
model_training.py
-----------------
Trains multiple classifiers, tunes Random Forest, saves best model.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import preprocess_pipeline


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")


def train_baseline_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:25s} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return results


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    print("\n🔍 Tuning Random Forest via GridSearchCV...")
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy",
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"\n✅ Best Parameters: {grid.best_params_}")
    print(f"✅ Best CV Score: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    return acc


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"\n💾 Saved best model to: {BEST_MODEL_PATH}")


def main():
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    train_baseline_models(X_train, y_train)
    best_rf = tune_random_forest(X_train, y_train)
    evaluate_model(best_rf, X_test, y_test)
    save_model(best_rf)


if __name__ == "__main__":
    main()