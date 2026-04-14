from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv("../winequality-red.csv", sep=";")
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]


def grid_search():
    param_grid = {
        "n_estimators": [10, 155, 300],
        "max_depth": [2, 15, 30],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 10, 20],
        "max_features": [0.1, 0.5, 1],
        "bootstrap": [True, False],
        "criterion": ["entropy", "gini"],
        "class_weight": ["balanced", None],
        "max_leaf_nodes": [10, 100, 200],
        "min_impurity_decrease": [0, 0.05, 0.1],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    total = len(combinations)

    print(f"Total combinations: {total}")
    print("-" * 50)

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{idx + 1}/{total}] {params}")

        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_score = scores.mean()

        results.append({**params, "score": mean_score})
        print(f"  -> Score: {mean_score:.4f}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 50)
    print("TOP 5 CONFIGURATIONS:")
    for i, res in enumerate(results[:5]):
        print(f"{i + 1}. Score: {res['score']:.4f}")
        params_only = {k: v for k, v in res.items() if k != "score"}
        print(f"   {params_only}")

    print("\n" + "=" * 50)
    print(f"BEST: {results[0]['score']:.4f}")

    best_params_only = {k: v for k, v in results[0].items() if k != "score"}
    print(best_params_only)

    return best_params_only


if __name__ == "__main__":
    best = grid_search()
