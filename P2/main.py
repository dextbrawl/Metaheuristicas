import random

import CreatePopulation
import crossing
import individuals
import replacement
import Selection

mutation_prob = 0.3
cross_prob = 0.7

if __name__ == "__main__":
    population_size = 18
    print("flag")
    population = CreatePopulation.CreateSequentialPopulation(population_size, 0.15, 100)
    max_iter = 100
    i = 0
    while i < max_iter:
        print("while")
        selected = Selection.TournamentSelection(population)
        print("seleccionados")
        children = []
        fathers = []
        for j in range(0, len(selected), 2):
            print()
            print("for1")
            n_points = 4
            c_prob = random.uniform(0.0, 1.0)
            if c_prob < cross_prob:
                print("cruza")
                child = crossing.n_point_crossing(
                    selected[j], selected[j + 1], n_points
                )
                fathers.append(selected[j])
                fathers.append(selected[j + 1])
                children.append(child)
        for j in range(0, len(children)):
            print("muta")
            m_prob = random.uniform(0.0, 1.0)
            if m_prob < mutation_prob:
                children[j].mutate()

        old_population = population
        population = replacement.crowdingReplacement(old_population, fathers, children)

        print("La populasini gusini ha quedado así:")
        print("----LISTA DE PANCHITOS----")
        for j in population:
            j.features()
        i += 1
