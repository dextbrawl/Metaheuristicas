import random
import numpy as np

import CreatePopulation
import crossing
import individuals
import replacement
import Selection
import model

mutation_prob = 0.5
cross_prob = 0.5

MAX_MUTATION_PROB, MIN_MUTATION_PROB = 0.8, 0.1
MAX_CROSSING_PROB, MIN_CROSSING_PROB = 0.9, 0.2
STAGNATION_THRESHOLD = 0.1

def populationVariance(population):
    vectors = np.array([CreatePopulation.NormaliseIndividual(ind) for ind in population])
    return np.mean(np.var(vectors, axis=0))

def changeProbabilities(mutation_prob, cross_prob, population):
    
    scores = [ind.score for ind in population]
    mean = np.mean(scores)
    variance = populationVariance(population)
    stagnated = False

    # Hay un estancamiento, aumentamos mutación y disminuimos cruce
    if variance < STAGNATION_THRESHOLD:
        mutation_prob = min(mutation_prob + 0.1, MAX_MUTATION_PROB)
        cross_prob = max(cross_prob - 0.1, MIN_CROSSING_PROB)
        stagnated = True
        print("Explorando")
    else:   # No hay estancamiento, seguimos intensificando
        print("Intensificando")
        mutation_prob = max(mutation_prob - 0.1, MIN_MUTATION_PROB)
        cross_prob = min(cross_prob + 0.1, MAX_CROSSING_PROB)
        stagnated = False
    
    print(f"Media de score: {mean}, Varianza de parámetros: {variance}\nProbabilidades actualizadas --> Mutación: {mutation_prob}, Cruce: {cross_prob}")
    with open("prob_data.csv", "a") as f:
        f.write(f"\n{mean},{variance},{mutation_prob},{cross_prob}")
    
    return mutation_prob, cross_prob, stagnated

if __name__ == "__main__":
    
    print(f"Prueba: {model.evaluate_solution(individuals.Individual(250, 43, 15, 35, "sqrt", False, 1, 0, 155, 0, 42, 0))}")
    
    population_size = 24
    population = CreatePopulation.CreateSequentialPopulation(population_size, 0.5, 150)
    print("Creada la population")
    
    # Variable para ver si el algoritmo está atrapado en un óptimo local
    stagnated = False
    
    # FIchero para recopilar datos
    with open("prob_data.csv", "w") as f:
        f.write("media_score,varianza_params,prob_mut,prob_cross")
    
    print("Población Inicial:")
    for j in population:
        j.features()
    
    print(f"\nProbabilidad de mutación y cruce iniciales: {mutation_prob}, {cross_prob}")
    
    max_iter = 50
    i = 0
    while i < max_iter:
        selected = Selection.TournamentSelection(population)
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

        for j in range(0, len(children)):
            m_prob = random.uniform(0.0, 1.0)
            if m_prob < mutation_prob:
                children[j].mutate()

                

        old_population = population
        if stagnated:
            population = replacement.replaceWorst(population, 5)
        else:
            population = replacement.replaceWorstWithChildren(population, children)
        

        print("La populasini gusini ha quedado así:")
        for j in population:
            j.features()
        i += 1
        
        mutation_prob, cross_prob, stagnated = changeProbabilities(mutation_prob, cross_prob, population)
        print("")

    max_score = 0

    for i in population:
        if(max_score < i.score):
            max_score = i.score
            final_individual = i
            
    
    print(f"El mejor individuo, con puntuación: {max_score}")
    final_individual.features()