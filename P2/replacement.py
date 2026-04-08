import numpy as np
import individuals as ind
import CreatePopulation as crt
import model as mod

def crowdingReplacement(population, fathers, children):
    
    assert(len(fathers) == (len(children)*2))
    
    # Los índices 0 y 1 de los padres corresponde al hijo 0 y así
    
    children_count = 0
    opponents = []
    
    for i in range(0, fathers, 2):
        
        # Escogemos al padre más parecido para reemplazarlo
        
        if(crt.PrimeDistance(fathers[i], children[children_count]) < crt.PrimeDistance(fathers[i+1], children[children_count])):
            opponents.append(fathers[i])
        else:
            opponents.append(fathers[i+1])
        
        children_count = children_count+1
    
    # Reemplazamos al padre más parecido si es peor que su hijo
    
    for i in range(opponents):
        
        if(mod.evaluate_solution(children[i]) > mod.evaluate_solution(opponents[i])):
            children[i].position = opponents[i].position
            population[opponents[i].position] = children[i]
    
    return population