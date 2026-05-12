import numpy as np
import random
from typing import List, Tuple, Optional
#from P2.Algoritmo_Genetico import individuals
from individual import Individual  
import prueba as modelo

class Population:
    
    def __init__(self, size: int, numPoints: int = 20, limits: Tuple[float, float] = (-20.0, 20.0), model = None):
    
        self.size = size
        self.numPoints = numPoints
        self.limits = limits
        self.model = model
        self.individuals = []
        self.bestFitnessHistory = []
        self.avgFitnessHistory = []
    
    def initializeRandom(self):
        
        self.individuals = []
        
       
        for i in range(self.size):
            ind = Individual(
                numPoints=self.numPoints,
                limits=self.limits,
                model=self.model
            )
            self.individuals.append(ind)
            
            
        
    def evaluateAll(self):
        """
        Evalúa el fitness de todos los individuos de la población
        """
        for ind in self.individuals:
            ind.computeFitness(self.model)
    
    def getBest(self) -> Optional[Individual]:
        if not self.individuals:
            return None
        
        self.evaluateAll()
        best = min(self.individuals, key=lambda ind: ind.fitness)
        return best
    
    
    def tournamentSelection(self, tournamentSize: int = 3) -> Individual:
        tournamentIndices = random.sample(range(len(self.individuals)), tournamentSize)
        tournament = [self.individuals[i] for i in tournamentIndices]
        
        winner = min(tournament, key=lambda ind: ind.fitness)
        return winner
    

    def mutation(self, pM, mRate, eliteSize=2):
        """
        pM: Probabilidad de que un individuo mude (0.0 a 1.0).
        mRate: Intensidad del cambio en las coordenadas de los puntos.
        eliteSize: número de individuos al principio de la lista que NO se mutan.
        """
        for i in range(eliteSize, len(self.individuals)):
            if random.random() < pM:
                ind = self.individuals[i]
                
                noise = np.random.uniform(-mRate, mRate, size=ind.points.shape)
                ind.points += noise
                
                ind.points = np.clip(ind.points, self.limits[0], self.limits[1])
                
                ind.classes = None
                ind.fitness = None

    def crossing(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if not parent1.pairs:
            parent1.smartPairing()
        if not parent2.pairs:
            parent2.smartPairing()
        
        child1 = Individual(numPoints=parent1.numPoints, limits=parent1.limits, model=parent1.model)
        child2 = Individual(numPoints=parent1.numPoints, limits=parent1.limits, model=parent1.model)
        
        pairs1 = parent1.pairs
        pairs2 = parent2.pairs
        
        for pairIndex in range(len(pairs1)):
            if random.random() < 0.5:
                # Hijo1 recibe el par del padre1, Hijo2 recibe el par del padre2
                i1, j1 = pairs1[pairIndex]
                child1.points[i1] = parent1.points[i1].copy()
                child1.points[j1] = parent1.points[j1].copy()
                
                i2, j2 = pairs2[pairIndex]
                child2.points[i2] = parent2.points[i2].copy()
                child2.points[j2] = parent2.points[j2].copy()
            else:
                # Hijo1 recibe el par del padre2, Hijo2 recibe el par del padre1
                i1, j1 = pairs2[pairIndex]
                child1.points[i1] = parent2.points[i1].copy()
                child1.points[j1] = parent2.points[j1].copy()
                
                i2, j2 = pairs1[pairIndex]
                child2.points[i2] = parent1.points[i2].copy()
                child2.points[j2] = parent1.points[j2].copy()
        
        child1.classes = None
        child1.fitness = None
        child1.pairs = []
        
        child2.classes = None
        child2.fitness = None
        child2.pairs = []
        
        return child1, child2

    def getStatistics(self) -> dict:
        self.evaluateAll()
        fitnesses = [ind.fitness for ind in self.individuals]
        
        stats = {
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'avg': np.mean(fitnesses),
            'std': np.std(fitnesses)
        }
        return stats
    
    def showStatistics(self):
        """
        Muestra las estadísticas de la población por pantalla
        """
        stats = self.getStatistics()
        print("POBLACION:")
        print(f"  Mejor fitness (min):    {stats['min']:.4f}")
        print(f"  Peor fitness (max):     {stats['max']:.4f}")
        print(f"  Fitness promedio:          {stats['avg']:.4f}")
        print(f"  Desviación estandar:       {stats['std']:.4f}")
    
    def showBestIndividual(self):
        best = self.getBest()
        if best:
            print("TOP PANCHITO")
            best.showInfo()


if __name__ == "__main__":
    print("PRUEBA DE LA CLASE POPULATION")
    model = modelo.BlackBoxModel("blackbox_modelB.pkl")
    
    population = Population(
        size=20,              # 20 individuos
        numPoints=20,         # cada uno con 20 puntos (10 pares)
        limits=(-1.0, 1.0),   # rango [-1, 1] (ajustado a tu modelo)
        model=model
    )
    
    population.initializeRandom()
    population.showStatistics()
    population.showBestIndividual()
    
    print("\nSELECCION DE PADRES TORNEO")
    winner = population.tournamentSelection(tournamentSize=5)
    print(f"Ganador del torneo con fitness: {winner.fitness:.4f}")
