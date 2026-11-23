from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BLUE, ORANGE, GREEN = "#245983", "#b65020", "#156681"


def preprocess(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=1)
    minimum = x_train.min(axis=0)
    span = x_train.max(axis=0) - minimum

    scaled_train = (x_train - minimum) / span
    scaled_test = (x_test - minimum) / span
    scaled_mean = scaled_train.mean(axis=0)

    return {
        "V1": (x_train - mean, x_test - mean),
        "V2": (scaled_train - scaled_mean, scaled_test - scaled_mean),
        "V3": ((x_train - mean) / std, (x_test - mean) / std),
    }


def fix_signs(vectors):
    vectors = vectors.copy()
    for j in range(vectors.shape[1]):
        column = vectors[:, j]
        if column[np.argmax(np.abs(column))] < 0:
            vectors[:, j] *= -1
    return vectors


def fit_pca(x):
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (len(x) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return cov, values[order], fix_signs(vectors[:, order])


def save_table(table, filename):
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table.to_csv(RESULTS / filename, index=False, float_format="%.4f")


def save_plot(fig, filename):
    fig.savefig(RESULTS / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_data():
    plt.switch_backend("Agg")
    RESULTS.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "wine_train.csv")
    test = pd.read_csv(ROOT / "data" / "wine_test.csv")
    names = train.columns.drop("class").tolist()
    variants = preprocess(train[names].to_numpy(), test[names].to_numpy())
    return variants, train["class"].to_numpy(), test["class"].to_numpy(), names
