import random
import numpy as np


def Tournament(population,size): #Usamos 3 o 2
   
    rival = random.sample(population, size)
    winner = max(rival, key=lambda x: x[1])
    return winner[0]




def TournamentSelection(population, tournamentSize=3, elite=2):
    PopulationSorted = sorted(population, key=lambda x: x[1], reverse=True)
    
    #Elitismo god
    selected = [ind for ind, _ in PopulationSorted[:elite]]
    
    
    rest = len(population) - elite
    for _ in range(rest):
        parent = Tournament(population, tournamentSize)
        selected.append(parent)
    
    #Todos estos son los padres
    return selected 


