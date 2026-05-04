import numpy as np
import random
from typing import List, Tuple, Optional
from individual import Individual  

class Population:
    #Limite -20 a 20
    def __init__(self, size: int, numpairs: int = 10, bounds: Tuple[float, float] = (-20.0, 20.0), model = None):
      
        self.size = size #Numero de individuos
        self.numpairs = numpairs #Numero de pares de cada individuo
        self.bounds = bounds #Limites
        self.model = model
        self.individuals = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def randomPopulation(self):
    
        self.individuals = []
        
        for i in range(self.size):
            ind = Individual(
                numpairs=self.numpairs,
                bounds=self.bounds,
                model=self.model
            )
            self.individuals.append(ind)
            
            if (i + 1) % 10 == 0:
                print(f"Generado individuo {i+1}/{self.size}")
        
        print(f"\nPoblacion inicial de {len(self.individuals)} individuos")
        print(f"Cada individuo tiene {self.numpairs} pares de puntos")
    
    def evaluate(self):
    
        for ind in self.individuals:
            ind.computeFitness(self.model)
    
    def getBest(self) -> Optional[Individual]:
        
        if not self.individuals:
            return None
        
        self.evaluate()
        best = min(self.individuals, key=lambda ind: ind.fitness)
        return best
    


    def tournament(self, tournament_size: int = 3) -> Individual:
       
        
        tournament_indices = random.sample(range(len(self.individuals)), tournament_size)
        tournament = [self.individuals[i] for i in tournament_indices]
        
        # OJO CRACK ES MEJOR TENER MENOS FITNES
        winner = min(tournament, key=lambda ind: ind.fitness)
        return winner
    
    #Este metodo (y el print) ahora mismo no sirven de mucho pero luego para ver la convergencia seran god yo creo
    def getStatistics(self) -> dict:
       
        self.evaluate()
        fitnesses = [ind.fitness for ind in self.individuals]
        
        stats = {
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'avg': np.mean(fitnesses),
            'std': np.std(fitnesses)
        }
        return stats
    
    def print(self):
       
        stats = self.getStatistics()
        print("\n--- POBLACION CREADA: ---")
        print(f"Mejor fitness: {stats['min']:.4f}")
        print(f"Peor fitness: {stats['max']:.4f}")
        print(f"Fitness promedio: {stats['avg']:.4f}")
        print(f"Desviacion estandar: {stats['std']:.4f}")


#ESTE MAIN CREA UNA POBLACION DE TAMAÑO SIZE Y LO SACA POR PANTALLA
if __name__ == "__main__":
    import prueba as modelo
    
    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    print("\nCreando poblacion inicial...")
    #Size son 20 individuos con 10 pares cada uno
    population = Population(size=20,numpairs=10,model=model)
    
    population.randomPopulation()
    
    best = population.getBest()
    if best:
        print(f"\n--- PANCHITO ELITE --")
        print(f"Fitness: {best.fitness:.4f}")
        print(f"Distancia promedio entre pares: {best.getDistances():.4f}")
        print(f"Dispersion: {best.getDispersion():.4f}")