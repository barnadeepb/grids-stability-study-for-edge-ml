"""
Regenerates the result charts (SVG + PNG) from the already-saved
results/model_comparison.csv, without re-running any model training.

Use this after changing chart styling or the protection-budget constant in
run_experiment.py, so the charts stay in sync with the paper without an
hours-long full re-run.
"""

import os

import pandas as pd

from run_experiment import RESULTS_DIR, make_plots

if __name__ == "__main__":
    results = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    make_plots(results)
