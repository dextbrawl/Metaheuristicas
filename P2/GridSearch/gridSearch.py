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
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
        "class_weight": [0, 1],
        "max_leaf_nodes": [24, 80, 170],
        "min_impurity_decrease": [0.2, 0.5, 0.7],
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

        try:
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            mean_score = scores.mean()

            results.append({**params, "score": mean_score})
            print(f"  -> Score: {mean_score:.4f}")
        except Exception as e:
            print(f"  -> Error: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 50)
    print("TOP 5 CONFIGURATIONS:")
    for i, res in enumerate(results[:5]):
        print(f"{i + 1}. Score: {res['score']:.4f}")
        del res["score"]
        print(f"   {res}")

    print("\n" + "=" * 50)
    print(f"BEST: {results[0]['score']:.4f}")
    del results[0]["score"]
    print(results[0])

    return results[0]


if __name__ == "__main__":
    best = grid_search()
