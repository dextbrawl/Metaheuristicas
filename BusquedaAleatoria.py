import random
import numpy as np

def readSeries(filename) -> list: # Función para crear la lista que define la serie
    data = np.loadtxt(filename).tolist()
    return data

def getBreakingPoints(n_points, k_segments): # Función de búsqueda aleatoria
    
    breaking_points = [0,n_points-1]

    for _ in range(k_segments -1):
        n = random.randint(0,n_points-1)
        
        while n in breaking_points:
            n = random.randint(0,n_points-1)
        
        breaking_points.append(n)
    
    breaking_points.sort()

    return breaking_points

def avgError(series: list, start_point, end_point): # Función para calcular el error medio de cada segmento
    

def randomSearch(series: list, k_segments):

    
