class Individual:
    def __init__(self, n_estimators, max_depth, min_samples_split, 
                 min_samples_leaf, max_features, bootstrap, criterion, 
                 class_weight, max_leaf_nodes, min_impurity_decrease, random_state):
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


