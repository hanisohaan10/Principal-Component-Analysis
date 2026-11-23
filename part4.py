import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import BLUE, GREEN, ORANGE, fit_pca, load_data, save_plot, save_table


def f_scores(x, y):
    classes = np.unique(y)
    mean = x.mean(axis=0)
    between = np.zeros(x.shape[1])
    within = np.zeros(x.shape[1])
    for c in classes:
        group = x[y == c]
        group_mean = group.mean(axis=0)
        between += len(group) * (group_mean - mean) ** 2
        within += ((group - group_mean) ** 2).sum(axis=0)
    return (between / (len(classes) - 1)) / (within / (len(y) - len(classes)))


def forward_selection(x, y, model, folds, max_k):
    selected = []
    for _ in range(max_k):
        best_score = -1
        for j in range(x.shape[1]):
            if j in selected:
                continue
            columns = sorted(selected + [j])
            scores = cross_val_score(model, x[:, columns], y, cv=folds, scoring="accuracy")
            score = scores.mean()
            if score > best_score:  # The first candidate wins a tie.
                best_score = score
                best_feature = j
        selected.append(best_feature)
    return selected


def part4(z_train, z_test, y_train, y_test, vectors, names):
    print("\nP4. PCA and feature selection")
    ks = [1, 2, 3, 5, 8]
    model = LogisticRegression(C=1.0, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(cv.split(z_train, y_train))
    scores = f_scores(z_train, y_train)
    ranking = np.argsort(-scores, kind="stable")
    selected = forward_selection(z_train, y_train, model, folds, max(ks))
    ranking_table = pd.DataFrame({"Rank": np.arange(1, len(names) + 1),
                                  "Feature": [names[j] for j in ranking], "F-score": scores[ranking]})
    save_table(ranking_table, "p4_f_scores.csv")

    baseline = model.fit(z_train, y_train).score(z_test, y_test)
    print(f"All-feature baseline = {baseline:.4f}")
    rows = []
    for k in ks:
        directions = vectors[:, :k]
        pca_acc = model.fit(z_train @ directions, y_train).score(z_test @ directions, y_test)
        filtered = ranking[:k]
        filter_acc = model.fit(z_train[:, filtered], y_train).score(z_test[:, filtered], y_test)
        wrapper = selected[:k]
        wrapper_acc = model.fit(z_train[:, wrapper], y_train).score(z_test[:, wrapper], y_test)
        rows.append([k, pca_acc, filter_acc, wrapper_acc])
        if k in (3, 8):
            print(f"k = {k} filter: {', '.join(names[j] for j in filtered)}")
            print(f"k = {k} wrapper: {', '.join(names[j] for j in wrapper)}")
            print(f"Features in common = {len(set(filtered) & set(wrapper))}")

    table = pd.DataFrame(rows, columns=["k", "PCA", "F-score", "SFS"])
    save_table(table, "p4_accuracy.csv")
    min_correct = round(baseline * len(y_test)) - 1
    for method in ["PCA", "F-score", "SFS"]:
        qualifying = table.loc[np.rint(table[method] * len(y_test)) >= min_correct, "k"]
        first_k = qualifying.iloc[0] if len(qualifying) else "none"
        print(f"First k within one test sample ({method}): {first_k}")

    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout="constrained")
    ax.axhline(baseline, color="#505050", ls="--", label="All 13 features")
    ax.plot(ks, table["PCA"], "o-", color=BLUE, ms=7, mfc="none", label="PCA")
    ax.plot(ks, table["F-score"], "s--", color=ORANGE, ms=5, mfc="none", label="F-score")
    ax.plot(ks, table["SFS"], "x-.", color=GREEN, ms=6, label="Forward selection")
    ax.set(title="Logistic regression test accuracy", xlabel="Dimensions retained (k)",
           ylabel="Test accuracy", xticks=ks)
    ax.legend()
    save_plot(fig, "p4_accuracy.png")


if __name__ == "__main__":
    variants, y_train, y_test, names = load_data()
    z_train, z_test = variants["V3"]
    _, _, vectors = fit_pca(z_train)
    part4(z_train, z_test, y_train, y_test, vectors, names)
