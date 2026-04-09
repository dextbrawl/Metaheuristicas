import numpy as np

import CreatePopulation as crt
import individuals as ind
import model as mod


def crowdingReplacement(population, fathers, children):

    if len(children) == 0:
        return population

    # Los índices 0 y 1 de los padres corresponde al hijo 0 y así

    children_count = 0
    opponents = []

    # Tomamos tantos padres como hijos tengamos (en pares)
    for i in range(0, len(children) * 2, 2):
        # Escogemos al padre más parecido para reemplazarlo

        if crt.distanceIdividuals(
            fathers[i], children[children_count]
        ) < crt.distanceIdividuals(fathers[i + 1], children[children_count]):
            opponents.append(fathers[i])
        else:
            opponents.append(fathers[i + 1])

        children_count = children_count + 1

    # Reemplazamos al padre más parecido si es peor que su hijo

    for i in range(len(opponents)):
        if children[i].score > opponents[i].score:
            children[i].position = opponents[i].position
            population[opponents[i].position] = children[i]
            print("Este hijo ha reemplazado a su padre: ")
            children[i].features()
            print("--->")
            opponents[i].features()

    return population

def worstReplacement(population, children):
    
    n_replacements = len(children)
    worst_individuals = []
    

    
    while(len(worst_individuals) < n_replacements):
        
        