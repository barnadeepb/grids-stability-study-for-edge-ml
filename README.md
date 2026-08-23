# Edge-Deployability Trade-offs for Power Grid Stability Classification

This project compares seven machine learning models on a power-grid
stability classification task -- not just on accuracy, but on the
three-way trade-off between **accuracy**, **inference latency**, and
**model size** that determines whether a model could realistically run
on resource-limited edge hardware (e.g. a substation-grade edge server)
rather than a full data center, and against a real industry timing
budget for protection-critical decisions.

## Why this trade-off, and not just accuracy

Most published comparisons in this space report a single number: "our
model reached X% accuracy." That is not the whole story for a real-time
grid protection use case, where a model may need to run close to the
equipment it monitors, on hardware with a fixed CPU, memory, and power
budget. A model that is more accurate but far larger or slower may be
the wrong choice if it cannot fit the deployment target -- or the
decision cycle.

**The industry timing budget used here:** IEC 61850-5 defines message
performance classes for substation automation. Trip-critical protection
messages (Type 1A, e.g. GOOSE trip commands) are expected to be
delivered end-to-end within roughly **4 milliseconds**. This project
measures every model's inference latency against that real budget,
rather than reporting latency numbers with nothing to compare them to.

## Models compared

| Model | Notes |
|---|---|
| Logistic Regression | Simplest baseline |
| Decision Tree | Single tree |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting (XGBoost) | Industry-standard method for tabular sensor/telemetry data |
| Multi-Layer Perceptron (Full) | Small neural network, larger hidden layers |
| Multi-Layer Perceptron (Compact) | Same neural network family, fewer hidden units |
| TabPFN (Tabular Foundation Model) | Pretrained tabular transformer (in-context learning, no traditional training); optional, see below |

Each model is wrapped in a scikit-learn pipeline (feature scaling +
model), and every measurement below times the *whole* pipeline, not
just the final classifier call.

**Why TabPFN is included at all:** it represents the current, actively
researched "modern" approach to tabular data (published in *Nature*,
Jan 2025; Prior Labs). Rather than leave the obvious "why didn't you
compare against something current?" question unanswered, this project
includes it deliberately -- as the high-accuracy, high-cost extreme of
the trade-off curve. See Results below for what that costs in practice.

## Dataset

[Electrical Grid Stability Simulated Data](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data),
UCI Machine Learning Repository.

- Simulates a 4-node star power grid (1 producer, 3 consumers) and
  records, per snapshot, whether the grid was stable or unstable.
- 10,000 rows, 12 numeric input features (reaction times, power values,
  price-elasticity coefficients per node), no missing values.
- Citation: Arzamasov, V. (2018). *Electrical Grid Stability Simulated
  Data* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5PG66. License: CC BY 4.0.

**Scope note:** this is simulated data, not live utility SCADA data.
Treat results as a methodology demonstration rather than a claim about
real-world grid telemetry.

## How the experiment is run

1. Load the dataset and drop the `stab` column (it directly determines
   the label we're predicting, so keeping it would leak the answer).
2. For each model, run a small hyperparameter search (5-fold
   cross-validation on a training split) so no model is handicapped by
   an arbitrary default setting. TabPFN is a pretrained foundation model
   with nothing meaningful to grid-search in this context, so this step
   is skipped for it.
3. Using the best hyperparameters found, retrain and evaluate each
   model across several random train/test splits, and report the mean
   and standard deviation of each metric -- this avoids drawing
   conclusions from a single lucky or unlucky split. Six of the seven
   models use 5 splits; TabPFN uses 2 (see the TabPFN section below for
   why).
4. For each model, measure:
   - **Accuracy / precision / recall / F1** on held-out test data.
   - **Inference latency**: time to make one prediction, averaged over
     many repeated calls (after a warm-up call).
   - **Model size**: size on disk of the trained pipeline (a proxy for
     memory/storage footprint).

## Results

Measured on the development machine described below (all values are
averages across the evaluation splits described above):

| Model | Accuracy | F1 | Latency | Size | Trained on |
|---|---|---|---|---|---|
| Logistic Regression | 0.795 | 0.831 | 1.4 ms | 2 KB | 8,000 rows |
| Decision Tree | 0.851 | 0.880 | 1.5 ms | 52 KB | 8,000 rows |
| Random Forest | 0.918 | 0.936 | 14.1 ms | 12.8 MB | 8,000 rows |
| **Gradient Boosting (XGBoost)** | **0.949** | **0.961** | **1.9 ms** | **465 KB** | 8,000 rows |
| Multi-Layer Perceptron (Full) | 0.946 | 0.957 | 1.4 ms | 79 KB | 8,000 rows |
| Multi-Layer Perceptron (Compact) | 0.914 | 0.933 | 1.4 ms | 16 KB | 8,000 rows |
| TabPFN (Tabular Foundation Model) | 0.968 | 0.975 | 125,589 ms | 207 MB | 5,000 rows (capped, see below) |

Full table with precision/recall and standard deviations:
[`results/model_comparison.csv`](results/model_comparison.csv).

**Reading these against the ~4 ms protection budget:**

- Five of the seven models (Logistic Regression, Decision Tree, XGBoost,
  both MLP variants) fit comfortably inside the budget.
- **Random Forest -- a common default choice in academic ML-for-grid
  papers -- exceeds the budget by roughly 3.5x**, and is simultaneously
  the largest model short of TabPFN (12.8 MB) and not even the most
  accurate practical option. On this evidence it is dominated on every
  axis by Gradient Boosting: less accurate, slower, and ~28x larger.
- **Gradient Boosting (XGBoost) is the best practical candidate found
  here**: highest accuracy among budget-compliant models, comfortably
  under the latency budget, and a modest 465 KB footprint.
- **TabPFN achieves the highest raw accuracy (96.8%) but exceeds the
  protection budget by roughly four orders of magnitude** (~125 seconds
  vs. a ~4 millisecond budget) and is ~16x larger on disk than Random
  Forest, the next-largest model (and over 450x larger than Gradient
  Boosting, the recommended practical model). Its accuracy advantage is real but
  irrelevant for a trip-critical decision cycle -- see below for why
  that gap is structural, not just a matter of optimizing the code.

Charts (SVG and PNG) for every column in the results table, plus two
trade-off views, are in `results/`:

- `metric_accuracy`, `metric_precision`, `metric_recall`, `metric_f1`,
  `metric_latency_us`, `metric_size_kb` -- one chart per metric, all
  models.
- `tradeoff_accuracy_vs_latency`, `tradeoff_accuracy_vs_size` -- accuracy
  plotted against each deployability cost, log scale, with the ~4 ms
  protection budget marked where relevant.

## Testing environment and its limits

All measurements are taken on the development machine actually running
this code (a 12-core laptop), **not** on real edge hardware. Two
concrete deployment-class devices motivate this work but were not
available for direct benchmarking:

- **Lenovo ThinkEdge SE350 V2** -- a compact, semi-rugged single-socket
  edge server (Intel Xeon-D, up to 16 cores, up to 100W), representative
  of a node deployed close to equipment (e.g. a substation).
- **Lenovo ThinkEdge SE455 V3** -- a larger rugged edge server (AMD EPYC
  8004, up to 64 cores, up to 225W, NEBS Level 3 rated), representative
  of a regional aggregation node.

The gap between the 12-core development machine and either target device
means **absolute latency numbers should not be read as deployment
guarantees**. The relative ordering between models (which is larger,
which is slower, by roughly how much) is the meaningful result, and is
expected to hold directionally across hardware tiers since all models
are evaluated under identical conditions.

No physical power/energy measurement is performed -- there is no
instrumented hardware available for this study. Model size on disk is
used as the practical footprint proxy instead.

## TabPFN: setup, its own CPU limit, and why that matters for the paper

TabPFN's reference implementation (the `tabpfn` PyPI package, Prior
Labs) is **not** a plain `pip install`-and-run library:

1. It requires a free account and a one-time license acceptance at
   https://ux.priorlabs.ai before it will download its pretrained
   weights, even for local/offline inference.
2. After accepting the license, an API key from that account must be set
   as the `TABPFN_TOKEN` environment variable. Without it, this script
   **skips TabPFN automatically** and runs everything else normally --
   no account is required to reproduce the other six models.
3. The library itself refuses full-speed CPU inference above 5,000
   training rows by default ("Running on CPU with more than 5000 samples
   is not allowed by default due to slow performance"). This project
   respects that limit rather than overriding it: TabPFN's training
   context is capped at 5,000 rows (vs. 8,000 for every other model),
   and this is recorded in the results table for transparency.
4. Given the measured cost (~125 seconds per single-row prediction on
   CPU, see Results), TabPFN uses 3 timing repeats and 2 evaluation
   splits rather than the 1,000 repeats / 5 splits used for the other
   models -- otherwise a full run would take hours. Run-to-run variance
   in the single-prediction timing was under 2%, so this reduced sample
   size does not materially weaken the latency conclusion.

**Why this belongs in the paper, not just this README:** a model that
requires outbound internet access and an account-bound token just to
load its weights is a poor fit for the air-gapped or tightly firewalled
networks common in OT/substation environments, independent of its
latency. Combined with the CPU sample cap and the measured ~125-second
single-prediction latency, TabPFN is disqualified for trip-critical
protection on multiple independent grounds -- not only speed.

```bash
# Optional, only needed to include TabPFN in the comparison:
export TABPFN_TOKEN="<your-api-key-from-ux.priorlabs.ai>"
```

Do not commit this token to any file. It is only ever read from the
environment variable at run time.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # on Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Running

```bash
python scripts/download_data.py
python scripts/run_experiment.py
```

The dataset is downloaded once into `data/` (not tracked in git -- see
`.gitignore`). Results are written to `results/` as described above.

## Reproducibility

Hyperparameter search uses a fixed seed; final evaluation is repeated
over fixed random seeds (5 seeds for most models, 2 for TabPFN -- see
above) and averaged. Re-running the scripts should reproduce the same
results table, with the exception of TabPFN's exact latency figure,
which depends on the host machine's CPU.

## License

The code in this repository is released under the [MIT License](LICENSE).
This covers only the code written here -- it does not extend to the
dataset (licensed separately under CC BY 4.0 by its original author, see
Dataset section above), to TabPFN's pretrained weights (subject to Prior
Labs' own license, accepted separately by anyone who chooses to use
that part of the comparison), or to the other third-party libraries
listed in `requirements.txt`, each of which is used under its own
license.
