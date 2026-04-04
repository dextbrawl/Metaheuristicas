import random
import individuals as ind

attr_list = [
    "n_estimators",
    "max_depth",
    "min_samples_split", 
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
    "class_weight",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "random_state"
]

def ind_to_list(ind: ind.Individual):
    retVal = []

    retVal.append(ind.n_estimators)
    retVal.append(ind.max_depth)
    retVal.append(ind.min_samples_split)
    retVal.append(ind.min_samples_leaf)
    retVal.append(ind.max_features)
    retVal.append(ind.bootstrap)
    retVal.append(ind.criterion)
    retVal.append(ind.class_weight)
    retVal.append(ind.max_leaf_nodes)
    retVal.append(ind.min_impurity_decrease)
    retVal.append(ind.random_state)

    return retVal

def list_to_ind(attr: list):
    retVal = ind.Individual(attr[0],attr[1],attr[2],attr[3],attr[4],attr[5],attr[6],attr[7],attr[8],attr[9],attr[10])

    return retVal

def n_point_crossing(parent1: ind.Individual, parent2: ind.Individual, n_points):
    split_points = sorted(random.sample(range(1, len(attr_list)), n_points - 1))

    heritage = []

    parent_bit = True
    for i in range(1,len(attr_list)):
        heritage.append(parent_bit)
        if(i in split_points):
            parent_bit = not parent_bit

    parent1_attr = ind_to_list(parent1)
    parent2_attr = ind_to_list(parent2)

    child_attr = []

    for i in range(1,len(attr_list)):
        if heritage[i]:
            child_attr.append(parent1_attr[i])
        else:
            child_attr.append(parent2_attr[i])

    child = list_to_ind(child_attr)

    return child
    

def uniform_crossing(parent1: ind.Individual, parent2: ind.Individual):

    parent1_attr = ind_to_list(parent1)
    parent2_attr = ind_to_list(parent2)
    child_attr = []

    for i in range(1,len(attr_list)):
        if bool(random.randint(0,1)):
            child_attr.append(parent1_attr)
        else:
            child_attr.append(parent2_attr)

    child = list_to_ind(child_attr)

    return child


