import random

import CreatePopulation
import crossing
import individuals
import replacement
import Selection

mutation_prob = 0.2
cross_prob = 0.8

if __name__ == "__main__":
    population_size = 18
    population = CreatePopulation.CreateSequentialPopulation(population_size, 0.15, 100)
    print("Creada la population")
    max_iter = 50
    i = 0
    while i < max_iter:
        selected = Selection.TournamentSelection(population)
        print("Seleccionados los folladores vividores")
        children = []
        fathers = []
        for j in range(0, len(selected), 2):
            n_points = 4
            c_prob = random.uniform(0.0, 1.0)
            if c_prob < cross_prob:
                child = crossing.n_point_crossing(
                    selected[j], selected[j + 1], n_points
                )
                fathers.append(selected[j])
                fathers.append(selected[j + 1])
                children.append(child)
                print("ha habido sexo, hijo creado:")
                child.features()
        for j in range(0, len(children)):
            m_prob = random.uniform(0.0, 1.0)
            if m_prob < mutation_prob:
                children[j].mutate()
                print("ha habido mutation, hijo mutado: ")
                children[j].features()
                

        old_population = population
        population = replacement.crowdingReplacement(old_population, fathers, children)

        print("La populasini gusini ha quedado así:")
        print("----LISTA DE PANCHITOS----")
        for j in population:
            j.features()
        i += 1

    max_score = 0

    for i in population:
        if(max_score < i.score):
            max_score = i.score
            final_individual = i
            
    
    print(f"El mejor individuo, con puntuación: {max_score}")
    final_individual.features()