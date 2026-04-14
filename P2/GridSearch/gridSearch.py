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
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [8, 15],
        "min_samples_leaf": [4, 15],
        "max_features": ["log2"],
        "bootstrap": [True],
        "criterion": ["entropy"],
        "class_weight": [1],
        "max_leaf_nodes": [125],
        "min_impurity_decrease": [0.05],
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
