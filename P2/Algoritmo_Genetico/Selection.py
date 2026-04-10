import random

import numpy as np

import model as mod


def Tournament(population, size):
    rival = random.sample(population, size)
    scores = [(ind, ind.score) for ind in rival]
    winner = max(scores, key=lambda x: x[1])
    return winner[0]


def TournamentSelection(population, tournamentSize=3, elite=2):
    scored_population = [(ind, ind.score) for ind in population]
    PopulationSorted = sorted(scored_population, key=lambda x: x[1], reverse=True)

    # Elitismo god
    selected = [ind for ind, _ in PopulationSorted[:elite]]

    rest = len(population) - elite
    for _ in range(rest):
        parent = Tournament(population, tournamentSize)
        selected.append(parent)

    # Todos estos son los padres
    return selected
