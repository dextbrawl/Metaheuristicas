import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# cargar dataset
data = pd.read_csv("winequality-red.csv", sep=";")

# convertir problema a clasificación binaria
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]


class Individual:
    def __init__(
        self,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        criterion,
        class_weight,
        max_leaf_nodes,
        min_impurity_decrease,
        random_state,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.class_weight = class_weight
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state


def evaluate_solution(ind):
    model = RandomForestClassifier(
        n_estimators=int(ind.n_estimators),
        max_depth=int(ind.max_depth),
        min_samples_split=int(ind.min_samples_split),
        min_samples_leaf=int(ind.min_samples_leaf),
        max_features=float(ind.max_features),
        bootstrap=bool(ind.bootstrap),
        criterion="gini" if ind.criterion == 0 else "entropy",
        class_weight=None if ind.class_weight == 0 else "balanced",
        max_leaf_nodes=int(ind.max_leaf_nodes),
        min_impurity_decrease=float(ind.min_impurity_decrease),
        random_state=int(ind.random_state),
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean()


if __name__ == "__main__":
    params_obj = Individual(10, 2, 2, 1, 0.1, 0, 0, 0, 10, 0, 42)
    print(f"Accuracy promedio: {evaluate_solution(params_obj)}")
