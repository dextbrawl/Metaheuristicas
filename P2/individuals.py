import random

from numpy.random.mtrand import rand

import model


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
        position,
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
        self.position = position
        self.elite = False
        self.score = model.evaluate_solution(self)

    def mutate(self):
        mutation_params = {
            "n_estimators": lambda: random.randint(10, 500),
            "max_depth": lambda: random.randint(1, 50),
            "min_samples_leaf": lambda: random.randint(1, 50),
            "max_features": lambda: random.choice(
                [random.uniform(0.1, 1.0), "sqrt", "log2", None]
            ),
            "bootstrap": lambda: random.choice([True, False]),
            "criterion": lambda: random.choice([0, 1]),
            "class_weight": lambda: random.choice([0, 1]),
            "max_leaf_nodes": lambda: random.randint(10, 200),
            "min_impurity_decrease": lambda: random.uniform(0, 0.1),
            "random_state": lambda: random.randint(0, 1000),
        }

        param_name = random.choice(list(mutation_params.keys()))
        new_param_value = mutation_params[param_name]()
        setattr(self, param_name, new_param_value)
        self.score = model.evaluate_solution(self)

    def features(self):
        print(
            f"[{self.n_estimators}, {self.max_depth}, {self.min_samples_split}, {self.min_samples_leaf}, {self.max_features}, {self.bootstrap}, {self.criterion}, {self.class_weight}, {self.max_leaf_nodes}, {self.min_impurity_decrease}, {self.random_state}]"
        )
