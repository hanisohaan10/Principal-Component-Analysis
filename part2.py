import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, GREEN, ORANGE, fit_pca, load_data, save_plot, save_table


def reconstruction_error(x, directions):
    scores = x @ directions
    reconstructed = scores @ directions.T
    return np.sum((x - reconstructed) ** 2) / (len(x) - 1)


def part2(z_train, z_test, cov, values, vectors):
    print("\nP2. Variance and reconstruction")
    ks = np.arange(1, len(values) + 1)
    ratios = values / values.sum()
    cumulative = ratios.cumsum()
    elbow = ks[ratios >= 0.10][-1]
    levels = [0.80, 0.90, 0.95]
    for level in levels:
        print(f"k at {level:.2f} = {ks[cumulative >= level][0]}")
    print(f"Elbow = {elbow}, cumulative variance there = {cumulative[elbow - 1]:.4f}")
    print(f"Kaiser count = {np.count_nonzero(values > np.trace(cov) / len(values))}")

    train_error = [reconstruction_error(z_train, vectors[:, :k]) for k in ks]
    test_error = [reconstruction_error(z_test, vectors[:, :k]) for k in ks]
    tail_sums = [values[k:].sum() for k in ks]
    residual = np.max(np.abs(np.array(train_error) - tail_sums))
    print(f"Max reconstruction identity residual = {residual:.4e}")
    table = pd.DataFrame({"k": ks, "r_k": ratios, "R(k)": cumulative,
                          "J_train": train_error, "J_test": test_error})
    save_table(table, "p2_variance.csv")

    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
    ax.bar(ks, ratios, color=BLUE, label="Individual share")
    ax.axvline(elbow, color=ORANGE, ls="--", label=f"Elbow (k = {elbow})")
    ax.set(title="Variance of each component", xlabel="Component j",
           ylabel="Explained variance ratio", xticks=ks)
    ax.legend()
    save_plot(fig, "p2_scree.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
    ax.plot(ks, cumulative, "o-", color=BLUE, label="Cumulative share")
    for level, color, style in zip(levels, [GREEN, ORANGE, "#755a88"], ["--", "-.", ":"]):
        ax.axhline(level, color=color, ls=style, label=f"{level:.0%}")
    ax.set(title="Cumulative variance", xlabel="Components retained (k)",
           ylabel="Cumulative variance ratio", xticks=ks, ylim=(0.30, 1.04))
    ax.legend(loc="lower right", ncol=2)
    save_plot(fig, "p2_cumulative.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.2), layout="constrained")
    ax.plot(ks, train_error, "o-", color=BLUE, label="Train")
    ax.plot(ks, test_error, "x--", color=ORANGE, label="Test")
    ax.set(title="Reconstruction error", xlabel="Components retained (k)",
           ylabel="Squared error / (n − 1)", xticks=ks)
    ax.legend()
    save_plot(fig, "p2_reconstruction.png")


if __name__ == "__main__":
    variants, _, _, _ = load_data()
    z_train, z_test = variants["V3"]
    cov, values, vectors = fit_pca(z_train)
    part2(z_train, z_test, cov, values, vectors)
