import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier

from common import BLUE, GREEN, ORANGE, fit_pca, load_data, save_plot, save_table


def part3(variants, y_train, y_test, names):
    print("\nP3. Effect of preprocessing")
    titles = {"V1": "Centred only", "V2": "Min–max", "V3": "Z-score"}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, variant in zip(axes, variants):
        train, test = variants[variant]
        cov, values, vectors = fit_pca(train)
        train_2d = train @ vectors[:, :2]
        test_2d = test @ vectors[:, :2]
        model = KNeighborsClassifier(n_neighbors=5).fit(train_2d, y_train)
        r1, r2 = (values / values.sum())[:2]
        top_feature = names[np.argmax(np.abs(vectors[:, 0]))]
        rows.append([variant, r1, r2, r1 + r2, top_feature, model.score(test_2d, y_test)])

        if variant == "V1":
            variances = np.diag(cov)
            for i in np.argsort(-variances)[:3]:
                print(f"Raw sample variance of {names[i]} = {variances[i]:.4f}")

        for c, color in enumerate([BLUE, ORANGE, GREEN]):
            train_class = train_2d[y_train == c]
            test_class = test_2d[y_test == c]
            ax.scatter(train_class[:, 0], train_class[:, 1], color=color, marker="o",
                       s=24, alpha=0.70, edgecolors="none")
            ax.scatter(test_class[:, 0], test_class[:, 1], color=color, marker="x",
                       s=37, linewidths=1.25)
        ax.set(title=f"{variant} · {titles[variant]}", xlabel="PC1 score", ylabel="PC2 score")

    classes = [Line2D([], [], marker="s", ls="", color=color, label=f"Class {c}")
               for c, color in enumerate([BLUE, ORANGE, GREEN])]
    markers = [Line2D([], [], marker=mark, ls="", color="#303030", label=label)
               for mark, label in [("o", "Train"), ("x", "Test")]]
    fig.suptitle("First two principal component scores")
    fig.legend(handles=classes + markers, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    save_plot(fig, "p3_projections.png")
    table = pd.DataFrame(rows, columns=["Variant", "r1", "r2", "r1 + r2", "Top PC1 feature", "5-NN accuracy"])
    save_table(table, "p3_preprocessing.csv")


if __name__ == "__main__":
    variants, y_train, y_test, names = load_data()
    part3(variants, y_train, y_test, names)
