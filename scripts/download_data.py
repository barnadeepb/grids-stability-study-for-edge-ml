"""
Downloads the "Electrical Grid Stability Simulated Data" dataset from the
UCI Machine Learning Repository and saves it locally under data/.

Dataset citation:
    Arzamasov, V. (2018). Electrical Grid Stability Simulated Data.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5PG66
    License: CC BY 4.0

The dataset simulates a 4-node star power grid (1 producer, 3 consumers)
and records, for each simulated snapshot, whether the grid was stable or
unstable. It is not live utility SCADA data -- see the README for how
this limitation is handled.
"""

import os
import urllib.request

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/"
    "Data_for_UCI_named.csv"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "grid_stability.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already present at {OUTPUT_PATH}, skipping download.")
        return

    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
