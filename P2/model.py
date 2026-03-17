import pandas as pd
import numpy as np
import random
import individuals as ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# cargar dataset
data = pd.read_csv("winequality-red.csv")
# convertir problema a clasificación binaria
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]

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
        random_state=int(ind.random_state)
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean()

#Crear una poblacion random de tamaño population
def CreateRandomPopulation(population):
    poblacion = []
    
    for _ in range(population):
        individuo = ind.Individual(
            n_estimators=random.randint(10, 300),
            max_depth=random.randint(2, 30),
            min_samples_split=random.randint(2, 20),
            min_samples_leaf=random.randint(1, 20),
            max_features=round(random.uniform(0.1, 1.0), 2),
            bootstrap=random.choice([0, 1]),
            criterion=random.choice([0, 1]),
            class_weight=random.choice([0, 1]),
            max_leaf_nodes=random.randint(10, 200),
            min_impurity_decrease=round(random.uniform(0, 0.1), 3),
            random_state=random.randint(1, 1000)
        )
        poblacion.append(individuo)
    return poblacion

#A cada individuo le asigna su puntuacion, luego lo ordenaremos esto   
class Population:
    def __init__(self, individual):
        self.IndividualAndItScore = []
        
        for ind in individual:
            score = evaluate_solution(ind)
            self.IndividualAndItScore.append((ind, score))

if __name__ == "__main__":
    individuos_aleatorios = CreateRandomPopulation(20)
    
    
    p= Population(individuos_aleatorios)
    print(f"Población creada con {len(p.IndividualAndItScore)} individuos")
    