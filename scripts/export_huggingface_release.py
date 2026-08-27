"""
Trains and serialises the paper's recommended budget-compliant model (the
"Full" Multi-Layer Perceptron -- see Section V-B/E of the paper for why it,
not the higher-profile Gradient Boosting model, is the practical pick when
accuracy is statistically tied) as a standalone artifact for release.

This is a research/reproducibility artifact, not a certified or
production-ready component -- see MODEL_CARD.md in the output folder for
scope and limitations. Output goes to huggingface_release/, which is
excluded from the public GitHub repo (see .gitignore) pending upload to
Hugging Face once credentials are available.
"""

import json
import os

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "grid_stability.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "huggingface_release")

PRIMARY_SEED = 42
TEST_SIZE = 0.2

# Best hyperparameters found by the original grid search in run_experiment.py
# (see results/model_comparison.csv, row "Multi-Layer Perceptron (Full)"),
# reused here rather than re-searched, so this artifact matches the exact
# configuration the paper reports on.
BEST_PARAMS = {"alpha": 0.001, "hidden_layer_sizes": (64, 32)}


def load_dataset():
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=["stab", "stabf"])
    labels = (df["stabf"] == "unstable").astype(int)
    return features, labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=PRIMARY_SEED
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            max_iter=3000, early_stopping=True, random_state=PRIMARY_SEED, **BEST_PARAMS
        )),
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    model_path = os.path.join(OUTPUT_DIR, "grid_stability_mlp_full.joblib")
    joblib.dump(pipeline, model_path)

    metadata = {
        "model": "Multi-Layer Perceptron (Full)",
        "sklearn_pipeline_steps": ["StandardScaler", "MLPClassifier"],
        "hyperparameters": {**BEST_PARAMS, "max_iter": 3000, "early_stopping": True},
        "trained_on_n_samples": len(X_train),
        "input_features": list(X.columns),
        "label": "1 = unstable, 0 = stable (derived from sign of the dataset's 'stab' column)",
        "held_out_test_metrics": metrics,
        "training_dataset": "UCI Electrical Grid Stability Simulated Data (Arzamasov, 2018), CC BY 4.0",
        "random_seed": PRIMARY_SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Held-out test metrics: {metrics}")


if __name__ == "__main__":
    main()
