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

def replaceWorst(population, n):

    sorted_pop = sorted(population, key=lambda ind: ind.score)
    worst = sorted_pop[:n]

    for worst_ind in worst:
        new_ind = crt.CreateRandomIndividual(len(population))
        new_ind.position = worst_ind.position
        population[worst_ind.position] = new_ind

    return population

def replaceWorstWithChildren(population, children, min_distance=0.15):

    if not children:
        return population

    sorted_pop = sorted(population, key=lambda ind: ind.score)
    sorted_children = sorted(children, key=lambda ind: ind.score, reverse=True)

    # Limitamos al mínimo entre peores y hijos disponibles
    replacements = min(len(sorted_pop), len(sorted_children))

    for i in range(replacements):
        worst = sorted_pop[i]
        best_child = sorted_children[i]

        if best_child.score <= worst.score:
            break  # si el mejor hijo no supera al peor, los siguientes tampoco

        # Comprobamos que el hijo no sea casi idéntico a alguien ya en la población
        distance = crt.PrimeDistance(best_child, population)
        if distance < min_distance:
            print(f"  [replaceWorst] Hijo descartado por distancia {distance:.3f} < {min_distance}")
            continue

        best_child.position = worst.position
        population[worst.position] = best_child

    return population
