import random
import Selection as sel
import CreatePopulation as pop
import individuals as ind
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# cargar dataset
data = pd.read_csv("winequality-red.csv")

# convertir problema a clasificación binaria
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]


# En una poblacion, cada individuo tiene su score.
class Population:
    def __init__(self, individual):
        self.IndividualAndItScore = []

        for ind in individual:
            score = evaluate_solution(ind)
            self.IndividualAndItScore.append((ind, score))


def evaluate_solution(ind: ind.Individual):
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
    PopulationSize = 20  # Parametro

    print("Formas de incializar nuestra poblacion:")
    print("  1. Random")
    print("  2. Secuencial")

    while True:
        try:
            metodo = int(input("\nElige un método (1 o 2): "))
            if metodo in [1, 2]:
                break
            else:
                print("Tiene que ser 1 o 2")
        except ValueError:
            print("Tiene que ser 1 o 2.")

    # Generar población según el método elegido
    if metodo == 1:
        individuos = pop.CreateRandomPopulation(PopulationSize)
        print(f"Población aleatoria creada con {len(individuos)} individuos")
    else:
        Min = 0.15
        MaxTry = 100
        individuos = pop.CreateSequentialPopulation(PopulationSize, Min, MaxTry)
        print(f"Población secuencial creada con {len(individuos)} individuos")
