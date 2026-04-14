import random
import sys

sys.path.append("..")
from Algoritmo_Genetico.individuals import Individual
from Algoritmo_Genetico.model import evaluate_solution


# Busqueda aleatoria.
# Se inicializa la mejor puntuacion a 0, y se generan individuos de manera aleatoria, devolviendo el mejor al final
def randomSearch(ind: Individual, n_iter):
    best_score = 0
    for _ in range(0, n_iter):
        ind.n_estimators = random.randint(10, 300)
        ind.max_depth = random.randint(2, 30)
        ind.min_samples_split = random.randint(2, 20)
        ind.min_samples_leaf = random.randint(1, 20)
        ind.max_features = random.uniform(0.1, 1)
        ind.bootstrap = random.choice([True, False])
        ind.criterion = random.choice(["gini", "entropy"])
        ind.class_weight = random.choice([None, "balanced"])
        ind.max_leaf_nodes = random.randint(10, 200)
        ind.min_impurity_decrease = random.uniform(0, 1)
        ind.random_state = random.randint(0, 1000)
        ind.score = evaluate_solution(ind)
        if best_score < ind.score:
            best_score = ind.score
            best_ind = ind
    return best_ind
