import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

import CreatePopulation as pop
import individuals as ind
import Selection as sel

# cargar dataset
data = pd.read_csv("winequality-red.csv", sep=";")

# convertir problema a clasificación binaria
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]


def evaluate_solution(ind: ind.Individual):
    max_features = ind.max_features
    if isinstance(max_features, str) or max_features is None:
        max_feat_param = max_features
    else:
        # Convierte a float por si acaso
        val = float(max_features)
        if val > 1:
            max_feat_param = int(val)  # número de características
        else:
            max_feat_param = val  # fracción (0.0 - 1.0)

    model = RandomForestClassifier(
        n_estimators=int(ind.n_estimators),
        max_depth=int(ind.max_depth),
        min_samples_split=int(ind.min_samples_split),
        min_samples_leaf=int(ind.min_samples_leaf),
        max_features=max_feat_param,
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
