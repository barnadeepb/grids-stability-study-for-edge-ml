"""
Compares several machine learning models on a power-grid stability
classification task, not just on accuracy but on the accuracy / inference
latency / model-size trade-off that matters if a model has to run on
resource-limited edge hardware (e.g. a substation-grade edge server)
rather than in a full data center.

What this script does:
    1. Loads the grid stability dataset (run download_data.py first).
    2. Splits it into training and test data.
    3. For each candidate model, runs a small hyperparameter search
       (cross-validated on the training data only) so no model is
       handicapped by a poorly chosen default setting.
    4. Re-trains and evaluates the best version of each model across
       several random train/test splits, to report stable, averaged
       numbers rather than a single lucky/unlucky split.
    5. For each model, measures:
         - classification quality (accuracy, precision, recall, F1)
         - inference latency (time to make one prediction, averaged
           over many repeated calls)
         - model size on disk (a proxy for memory/storage footprint)
    6. Saves a results table (CSV) and comparison charts (SVG) to the
       results/ folder.

Important scope note (see README for the full discussion):
    Latency and size are measured on the development machine used to run
    this script, not on real edge hardware. Treat the absolute numbers
    as illustrative and the *relative* ordering between models as the
    meaningful result.

Optional TabPFN comparison:
    One model in this comparison, TabPFN, is a pretrained tabular
    foundation model distributed by Prior Labs. Using it requires a free
    account and a one-time license acceptance at https://ux.priorlabs.ai,
    and an API token set as the TABPFN_TOKEN environment variable. If
    that variable is not set, this script skips TabPFN automatically and
    still runs everything else -- no account is required to reproduce
    the rest of the comparison.
"""

import os
import textwrap
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "grid_stability.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

PRIMARY_SEED = 42
EVAL_SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2
DEFAULT_N_LATENCY_RUNS = 1000

# TabPFN documents that CPU inference above this many training rows is
# disabled by default because it's impractically slow -- we respect that
# limit rather than override it, and treat the limit itself as part of
# the result (see README).
#
# A single CPU prediction was independently timed at ~143 seconds with a
# 5,000-row training context (see README for the measured numbers), with
# very low run-to-run variance (~1%). Given that cost, TabPFN uses far
# fewer repeated timing runs and evaluation splits than the other models
# -- otherwise a full run would take hours rather than minutes. This is
# a deliberate, documented deviation from the other models' protocol,
# not an oversight.
TABPFN_MAX_TRAIN_ROWS = 5000
TABPFN_N_LATENCY_RUNS = 3
TABPFN_EVAL_SEEDS = [0, 1]


def build_scaled_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tabpfn_available():
    return bool(os.environ.get("TABPFN_TOKEN", "").strip())


def build_tabpfn_pipeline():
    from tabpfn import TabPFNClassifier  # imported lazily: optional dependency path

    return build_scaled_pipeline(TabPFNClassifier(device="cpu"))


# Each entry describes one model to evaluate:
#   name             - full, non-abbreviated label used in the results table and charts
#   build_pipeline   - callable returning a fresh (scaler + model) pipeline
#   param_grid       - hyperparameter grid for GridSearchCV, or None to skip tuning
#   max_train_rows   - cap on training rows for this model, or None for no cap
#   n_latency_runs   - repeated single-prediction timings to average over
#   eval_seeds       - list of random seeds to average the final metrics over
#   enabled          - callable returning True if this model should be run at all
MODEL_SPECS = [
    dict(
        name="Logistic Regression",
        build_pipeline=lambda: build_scaled_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="Decision Tree",
        build_pipeline=lambda: build_scaled_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=PRIMARY_SEED)
        ),
        param_grid={"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]},
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="Random Forest",
        build_pipeline=lambda: build_scaled_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=PRIMARY_SEED)
        ),
        param_grid={"model__n_estimators": [50, 100, 200], "model__max_depth": [5, 10, None]},
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="Gradient Boosting (XGBoost)",
        build_pipeline=lambda: build_scaled_pipeline(
            XGBClassifier(
                eval_metric="logloss",
                random_state=PRIMARY_SEED,
                n_jobs=1,
            )
        ),
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 8],
            "model__learning_rate": [0.05, 0.1, 0.3],
        },
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="Multi-Layer Perceptron (Full)",
        build_pipeline=lambda: build_scaled_pipeline(
            MLPClassifier(max_iter=3000, early_stopping=True, random_state=PRIMARY_SEED)
        ),
        param_grid={"model__hidden_layer_sizes": [(64, 32), (100,)], "model__alpha": [0.0001, 0.001]},
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="Multi-Layer Perceptron (Compact)",
        build_pipeline=lambda: build_scaled_pipeline(
            MLPClassifier(max_iter=3000, early_stopping=True, random_state=PRIMARY_SEED)
        ),
        param_grid={"model__hidden_layer_sizes": [(8,), (16,)], "model__alpha": [0.0001, 0.001]},
        max_train_rows=None,
        n_latency_runs=DEFAULT_N_LATENCY_RUNS,
        eval_seeds=EVAL_SEEDS,
        enabled=lambda: True,
    ),
    dict(
        name="TabPFN (Tabular Foundation Model)",
        build_pipeline=build_tabpfn_pipeline,
        param_grid=None,  # pretrained foundation model: nothing meaningful to grid-search here
        max_train_rows=TABPFN_MAX_TRAIN_ROWS,
        n_latency_runs=TABPFN_N_LATENCY_RUNS,
        eval_seeds=TABPFN_EVAL_SEEDS,
        enabled=tabpfn_available,
    ),
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # "stab" is the raw equilibrium value that "stabf" (the label we want to
    # predict) is directly derived from (stable if stab <= 0). Keeping it as
    # an input feature would leak the answer, so it is dropped here.
    features = df.drop(columns=["stab", "stabf"])
    labels = (df["stabf"] == "unstable").astype(int)
    return features, labels


def tune_hyperparameters(pipeline, param_grid, X_train, y_train):
    if param_grid is None:
        return {}
    search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_


def measure_latency_microseconds(pipeline, X_sample, n_runs):
    single_row = X_sample.iloc[[0]]
    # Warm-up call, excluded from timing, so one-off setup cost doesn't
    # distort the measurement.
    pipeline.predict(single_row)

    start = time.perf_counter()
    for _ in range(n_runs):
        pipeline.predict(single_row)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1e6


def measure_model_size_kb(pipeline, tag):
    temp_path = os.path.join(RESULTS_DIR, f".tmp_{tag}.joblib")
    joblib.dump(pipeline, temp_path)
    size_kb = os.path.getsize(temp_path) / 1024
    os.remove(temp_path)
    return size_kb


def evaluate_once(build_pipeline, best_params, X, y, seed, max_train_rows, n_latency_runs):
    pipeline = build_pipeline().set_params(**best_params)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    if max_train_rows is not None and len(X_train) > max_train_rows:
        X_train = X_train.iloc[:max_train_rows]
        y_train = y_train.iloc[:max_train_rows]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "latency_us": measure_latency_microseconds(pipeline, X_test, n_latency_runs),
    }
    return metrics, pipeline, len(X_train)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y = load_dataset()

    # One fixed split, used only to select hyperparameters via cross-validation.
    X_train_primary, _, y_train_primary, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=PRIMARY_SEED
    )

    rows = []
    for spec in MODEL_SPECS:
        name = spec["name"]

        if not spec["enabled"]():
            print(f"Skipping {name} (not available in this environment -- see README).")
            continue

        print(f"Tuning {name} ...")
        tuning_train_X, tuning_train_y = X_train_primary, y_train_primary
        if spec["max_train_rows"] is not None and len(tuning_train_X) > spec["max_train_rows"]:
            tuning_train_X = tuning_train_X.iloc[: spec["max_train_rows"]]
            tuning_train_y = tuning_train_y.iloc[: spec["max_train_rows"]]

        best_params = tune_hyperparameters(
            spec["build_pipeline"](), spec["param_grid"], tuning_train_X, tuning_train_y
        )
        print(f"  best params: {best_params if best_params else '(defaults; not tuned)'}")

        seed_metrics = []
        fitted_pipeline_for_size = None
        trained_on_n_samples = None
        for seed in spec["eval_seeds"]:
            metrics, fitted_pipeline, n_train = evaluate_once(
                spec["build_pipeline"], best_params, X, y, seed,
                spec["max_train_rows"], spec["n_latency_runs"],
            )
            seed_metrics.append(metrics)
            fitted_pipeline_for_size = fitted_pipeline
            trained_on_n_samples = n_train

        size_kb = measure_model_size_kb(fitted_pipeline_for_size, tag=name.replace(" ", "_"))

        averaged = {
            metric: np.mean([m[metric] for m in seed_metrics]) for metric in seed_metrics[0]
        }
        stdev = {
            f"{metric}_std": np.std([m[metric] for m in seed_metrics]) for metric in seed_metrics[0]
        }

        row = {
            "model": name,
            **averaged,
            **stdev,
            "size_kb": size_kb,
            "trained_on_n_samples": trained_on_n_samples,
            "best_params": best_params,
        }
        rows.append(row)
        print(f"  accuracy={row['accuracy']:.3f} f1={row['f1']:.3f} "
              f"latency={row['latency_us']:.1f}us size={row['size_kb']:.1f}KB "
              f"(trained on {trained_on_n_samples} rows)")

    results = pd.DataFrame(rows)
    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved results table to {results_path}")

    make_plots(results)


# IEC 61850-5 defines message performance classes for substation automation;
# trip-critical protection messages (Type 1A, e.g. GOOSE trip commands) are
# expected to be delivered end-to-end within about 4 milliseconds. This is
# an industry timing budget, not a number we chose -- it lets the chart show
# which models could plausibly fit inside a real protection decision cycle,
# rather than reporting latency with nothing to compare it to.
PROTECTION_LATENCY_BUDGET_MS = 4.0
PROTECTION_LATENCY_BUDGET_LABEL = "IEC 61850-5 Type 1A protection budget (~4 ms)"

# Print-ready styling: a serif face to match IEEE body text, restrained sizes,
# and hairline chrome rather than heavy gridlines/borders.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "svg.fonttype": "none",  # keep text as text in the SVG, not paths
})

BAR_COLOR = "#2a78d6"  # one series -> one color (categorical slot 1)
BUDGET_COLOR = "#d03b3b"  # reserved status "critical" color, distinct from all model colors
VALUE_TEXT_COLOR = "#52514e"

# One fixed color + marker per model, reused across every chart that plots
# per-model identity (only the trade-off scatter charts need this -- bar
# charts are single-series, see BAR_COLOR above). Marker shape is a second,
# redundant channel so identity never depends on color alone.
MODEL_STYLE = {
    "Logistic Regression":                 dict(color="#2a78d6", marker="o"),
    "Decision Tree":                       dict(color="#eb6834", marker="s"),
    "Random Forest":                       dict(color="#1baf7a", marker="^"),
    "Gradient Boosting (XGBoost)":         dict(color="#eda100", marker="D"),
    "Multi-Layer Perceptron (Full)":       dict(color="#e87ba4", marker="v"),
    "Multi-Layer Perceptron (Compact)":    dict(color="#008300", marker="p"),
    "TabPFN (Tabular Foundation Model)":   dict(color="#4a3aa7", marker="*"),
}


def wrap_label(name, width=14):
    return "\n".join(textwrap.wrap(name, width=width))


def format_size(size_kb):
    if size_kb >= 1024:
        return f"{size_kb / 1024:.0f} MB"
    return f"{size_kb:.0f} KB"


# (title, column, std_column or None, axis label, use_log_scale, value formatter)
METRIC_CHARTS = [
    ("Classification Accuracy", "accuracy", "accuracy_std", "Accuracy", False, lambda v: f"{v * 100:.1f}%"),
    ("Precision", "precision", "precision_std", "Precision", False, lambda v: f"{v * 100:.1f}%"),
    ("Recall", "recall", "recall_std", "Recall", False, lambda v: f"{v * 100:.1f}%"),
    ("F1 Score", "f1", "f1_std", "F1 score", False, lambda v: f"{v * 100:.1f}%"),
    ("Inference Latency", "latency_ms", "latency_ms_std", "Latency (ms, log scale)", True, lambda v: f"{v:,.1f} ms"),
    ("Model Size on Disk", "size_kb", None, "Model size (KB, log scale)", True, format_size),
]


def save_figure(fig, filename_stem):
    svg_path = os.path.join(RESULTS_DIR, f"{filename_stem}.svg")
    png_path = os.path.join(RESULTS_DIR, f"{filename_stem}.png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def make_plots(results):
    results = results.copy()
    results["latency_ms"] = results["latency_us"] / 1000
    results["latency_ms_std"] = results["latency_us_std"] / 1000

    x_positions = np.arange(len(results))
    wrapped_names = [wrap_label(name) for name in results["model"]]

    for title, column, std_column, axis_label, use_log_scale, format_value in METRIC_CHARTS:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        errors = results[std_column] if std_column is not None else None
        bars = ax.bar(
            x_positions, results[column], yerr=errors, capsize=3,
            color=BAR_COLOR, width=0.6,
            error_kw=dict(elinewidth=0.8, ecolor="#0b0b0b"),
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(wrapped_names)
        ax.set_ylabel(axis_label)
        ax.set_title(title, loc="left", fontweight="bold")
        style_axes(ax)

        if use_log_scale:
            ax.set_yscale("log")
            top = results[column].max()
            ax.set_ylim(top=top * 6)  # headroom so bar-top labels never collide with the frame
        else:
            ax.set_ylim(0, 1.08)

        for bar, value in zip(bars, results[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * (1.15 if use_log_scale else 1.0) + (0 if use_log_scale else 0.015),
                format_value(value), ha="center", va="bottom", fontsize=7, color=VALUE_TEXT_COLOR,
            )

        if column == "latency_ms":
            ax.axhline(PROTECTION_LATENCY_BUDGET_MS, color=BUDGET_COLOR, linestyle="--", linewidth=1.0)
            ax.plot([], [], color=BUDGET_COLOR, linestyle="--", linewidth=1.0, label=PROTECTION_LATENCY_BUDGET_LABEL)
            ax.legend(loc="upper left", frameon=False, handlelength=1.5)

        fig.tight_layout()
        save_figure(fig, f"metric_{column.replace('_ms', '_us')}")

    # Trade-off views: accuracy plotted against the two deployability costs.
    # Identity comes from the legend (color + marker), never from in-plot text,
    # so points that sit close together never produce overlapping labels.
    for cost_column, cost_label, filename_stem, budget in [
        ("latency_ms", "Inference latency per prediction (ms, log scale)", "tradeoff_accuracy_vs_latency", PROTECTION_LATENCY_BUDGET_MS),
        ("size_kb", "Model size on disk (KB, log scale)", "tradeoff_accuracy_vs_size", None),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 3.6))

        for _, row in results.iterrows():
            style = MODEL_STYLE[row["model"]]
            ax.errorbar(
                row[cost_column], row["accuracy"], yerr=row["accuracy_std"],
                fmt=style["marker"], color=style["color"], markersize=8,
                markeredgecolor="#0b0b0b", markeredgewidth=0.5,
                ecolor="#0b0b0b", elinewidth=0.8, capsize=3,
                label=row["model"],
            )

        ax.set_xscale("log")
        ax.set_xlabel(cost_label)
        ax.set_ylabel("Classification accuracy")
        ax.set_title(f"Accuracy vs. {cost_label.split(' (')[0].lower()}", loc="left", fontweight="bold")
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
        style_axes(ax)

        handles, labels = ax.get_legend_handles_labels()
        if budget is not None:
            ax.axvline(budget, color=BUDGET_COLOR, linestyle="--", linewidth=1.0)
            budget_handle = plt.Line2D([], [], color=BUDGET_COLOR, linestyle="--", linewidth=1.0)
            handles.append(budget_handle)
            labels.append(PROTECTION_LATENCY_BUDGET_LABEL)

        ax.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22),
            ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.2,
        )

        fig.tight_layout()
        save_figure(fig, filename_stem)

    print(f"Saved charts (SVG + PNG) to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
